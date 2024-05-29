import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv, GCNConv
from torch_geometric.nn import (
    global_mean_pool,
)
from torch_geometric.nn import Set2Set
from torch_geometric.nn import BatchNorm
from config import SUPPORTED_ATOMS, SUPPORTED_EDGES, MAX_MOLECULE_SIZE, ATOMIC_NUMBERS
from utils import graph_representation_to_molecule, to_one_hot
from tqdm import tqdm
import numpy
from torch_geometric.data import Data, Batch
from config import DEVICE as device

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        self.encoder_embedding_size = 64
        self.edge_dim = 11
        self.latent_embedding_size = 121           
        self.num_edge_types = len(SUPPORTED_EDGES) 
        self.num_atom_types = len(SUPPORTED_ATOMS)
        self.max_num_atoms = MAX_MOLECULE_SIZE 
        self.decoder_hidden_neurons = 512
        self.INTERMEDIATE_EDGE_INDEX = torch.tensor([[i, j] for i in range(MAX_MOLECULE_SIZE) for j in range(MAX_MOLECULE_SIZE)], dtype=torch.long).t().contiguous().to(device)


        # Encoder layers
        self.conv1 = TransformerConv(feature_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.conv2 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.conv3 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        self.conv4 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)

        # Latent transform layers
        self.mu_transform = Linear(self.encoder_embedding_size, 
                                            self.latent_embedding_size)
        self.logvar_transform = Linear(self.encoder_embedding_size, 
                                            self.latent_embedding_size)

        # Decoder layers
        # --- Shared layers
        self.linear_1 = Linear(self.latent_embedding_size, self.decoder_hidden_neurons)
        self.linear_2 = Linear(self.decoder_hidden_neurons, self.decoder_hidden_neurons)

        # --- Atom decoding (outputs a matrix: (max_num_atoms) * (# atom_types + "none"-type))   
        self.atom_output_dim = self.num_atom_types + 1
        self.atom_decode_layer = Linear(self.decoder_hidden_neurons, self.atom_output_dim)

        # --- Edge decoding (outputs a triu tensor: (max_num_atoms*(max_num_atoms-1)/2*(#edge_types + 1) ))
        self.edge_output_dim = int(((self.max_num_atoms * (self.max_num_atoms - 1)) / 2) * (self.num_edge_types + 1))
        self.edge_decode = Linear(self.decoder_hidden_neurons, self.edge_output_dim)
        self.edge_conv_decoder = GCNConv(self.latent_embedding_size, self.edge_output_dim)

        

    def encode(self, x, edge_attr, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_attr).relu()



        # Latent transform layers
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)
        return mu, logvar

    def decode_graph(self, graph_z, edge_index):  
        """
        Decodes a latent vector into a continuous graph representation
        consisting of node types and edge types.
        """
        # Decode edge types

        # take the inner product
        norm_Z = torch.norm(graph_z)
        hat_A = torch.mm(graph_z, graph_z.t()) / (norm_Z ** 2)

        # weighted adjacency graph
        hat_A = hat_A + torch.ones_like(hat_A)

        hat_A.fill_diagonal_(0)

        # 1D edge weight array
        hat_A = hat_A[hat_A != 0]


        # Generate all possible edges (i, j) where i != j
        n = MAX_MOLECULE_SIZE

        out = self.edge_conv_decoder(graph_z, edge_index, edge_weight = hat_A)

        edge_logits = torch.mean(out, dim=0)
        

        return edge_logits
    

    def decode_atoms(self, z):
        # Decode atom types
        z = self.linear_1(z).relu()
        z = self.linear_2(z).relu()
        z = self.atom_decode_layer(z)

        return z

    def decode_edges(self, z):
        # Reshape Z to represent a batch of 6 matrices of size 20 x 10
        
        z = z.view(-1, MAX_MOLECULE_SIZE, self.latent_embedding_size)

        # Compute Z*Z^T for each matrix in the batch
        #TODO is there a batch triu option?
        ZZT = torch.bmm(z, z.transpose(1, 2)) 

        # Compute the Frobenius norm squared for each matrix in the batch
        frobenius_norm_sq = torch.sum(z ** 2, dim=(1, 2), keepdim=True)  # Shape: (6, 1, 1)

        # Normalize ZZ^T by the Frobenius norm squared
        normalized_ZZT = ZZT / frobenius_norm_sq  # Broadcasting to (6, 20, 20)

        # Create a matrix of ones of the same size as each ZZ^T
        ones_matrix = torch.ones_like(ZZT)  # Shape: (6, 20, 20)

        # Compute the final result
        hat_A = normalized_ZZT + ones_matrix


        # Create a mask for the diagonal elements
        batch_size, matrix_size, _ = hat_A.shape
        mask = torch.eye(matrix_size).bool().unsqueeze(0).expand(batch_size, -1, -1)

        # Set the diagonal elements to 0
        hat_A[mask] = 0

        # max it a num_batch * num_edges matrix
        hat_A =  torch.reshape(hat_A, (-1, MAX_MOLECULE_SIZE * MAX_MOLECULE_SIZE))

        data_list = [Data(x=z[i], edge_index = self.INTERMEDIATE_EDGE_INDEX, edge_weight=hat_A[i]) for i in range(len(hat_A))]

        batch = Batch.from_data_list(data_list)


        #error here
        edge_logits = self.edge_conv_decoder(batch.x, edge_index = batch.edge_index, edge_weight = batch.edge_weight)


        # take the average
        edge_logits = edge_logits.view(-1, MAX_MOLECULE_SIZE, self.edge_output_dim)
        edge_logits = torch.mean(edge_logits, dim=1)
        edge_logits = edge_logits.flatten()

        return edge_logits


    def decode(self, z):
        triu_logits = []
        # Iterate over molecules in batch
        node_logits = self.decode_atoms(z).flatten()
        triu_logits = self.decode_edges(z)


        return triu_logits, node_logits


    def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_attr, edge_index, batch_index): # batch index tells us where each node is.
        # Encode the molecule
        mu, logvar = self.encode(x, edge_attr, edge_index, batch_index)
        # Sample latent vector (per atom)
        z = self.reparameterize(mu, logvar)
        #ends = self.find_ends(batch_index)
        # Decode latent vector into original molecule
        triu_logits, node_logits = self.decode(z)

        return triu_logits, node_logits, mu, logvar
    
    # index of the last node in a given graph. 
    def find_ends(self, batch_index):
        end = 0
        batch_index = batch_index.detach().numpy()
        ends = []
        while end < len(batch_index) - 1:
            graph_num = batch_index[end]
            index = numpy.where(batch_index == graph_num)
            end = index[0][-1] + 1
            ends.append(end)
        

        return ends



    
    def sample_mols(self, num=10000):
        print("Sampling molecules ... ")

        n_valid = 0
        # Sample molecules and check if they are valid
        for _ in tqdm(range(num)):
            # Sample latent space
            z = torch.randn(self.max_num_atoms, self.latent_embedding_size)

            # Get model output (this could also be batched)
            dummy_batch_index = torch.Tensor([0]).int()
            ends = [20]
            
            triu_logits, node_logits = self.decode(z, dummy_batch_index, ends)

            # Reshape triu predictions 
            edge_matrix_shape = (int((MAX_MOLECULE_SIZE * (MAX_MOLECULE_SIZE - 1))/2), len(SUPPORTED_EDGES) + 1) 
            triu_preds_matrix = triu_logits.reshape(edge_matrix_shape)
            triu_preds = torch.argmax(triu_preds_matrix, dim=1)

            # Reshape node predictions
            node_matrix_shape = (MAX_MOLECULE_SIZE, (len(SUPPORTED_ATOMS) + 1)) 
            node_preds_matrix = node_logits.reshape(node_matrix_shape)
            node_preds = torch.argmax(node_preds_matrix[:, :9], dim=1)
            
            # Get atomic numbers 
            node_preds_one_hot = to_one_hot(node_preds, options=ATOMIC_NUMBERS)
            atom_numbers_dummy = torch.Tensor(ATOMIC_NUMBERS).repeat(node_preds_one_hot.shape[0], 1)
            atom_types = torch.masked_select(atom_numbers_dummy, node_preds_one_hot.bool())

            # Attempt to create valid molecule
            smiles, _ = graph_representation_to_molecule(atom_types, triu_preds.float())

            # A dot means disconnected
            if smiles and "." not in smiles:
                print("Successfully generated: ", smiles)
                n_valid += 1    
        return n_valid