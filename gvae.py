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

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        self.encoder_embedding_size = 64
        self.edge_dim = 11
        self.latent_embedding_size = 121            # must be a square number, so we can convert it to a matrix and take an inner product with its transpose.
        self.num_edge_types = len(SUPPORTED_EDGES) 
        self.num_atom_types = len(SUPPORTED_ATOMS)
        self.max_num_atoms = MAX_MOLECULE_SIZE 
        self.decoder_hidden_neurons = 512

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
        edge_output_dim = int(((self.max_num_atoms * (self.max_num_atoms - 1)) / 2) * (self.num_edge_types + 1))
        self.edge_decode = Linear(self.decoder_hidden_neurons, edge_output_dim)
        self.edge_conv_decoder = GCNConv(self.latent_embedding_size, edge_output_dim)

        

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

    def decode_graph(self, graph_z):  
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
        collapsed_tensor = hat_A[hat_A != 0]


        # Generate all possible edges (i, j) where i != j
        n = MAX_MOLECULE_SIZE
        edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).t().contiguous()

        #error here
        out = self.edge_conv_decoder(graph_z, edge_index, edge_weight = collapsed_tensor)

        # take the values of the top half. 



        edge_logits = torch.mean(out, dim=0)
        

        return edge_logits
    

    def decode_atoms(self, z):
        # Decode atom types
        z = self.linear_1(z).relu()
        z = self.linear_2(z).relu()
        z = self.atom_decode_layer(z)

        return z



    def decode(self, z, batch_index, ends):
        triu_logits = []
        # Iterate over molecules in batch
        start = 0
        node_logits = self.decode_atoms(z).flatten()

        for end in ends:

            graph_z = z[start:end]

            # Recover graph from latent vector
            edge_logits = self.decode_graph(graph_z)
            # Store per graph results
            triu_logits.append(edge_logits)
            start = end


        # Concatenate all outputs of the batch
        triu_logits = torch.cat(triu_logits)
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
        ends = self.find_ends(batch_index)
        # Decode latent vector into original molecule
        triu_logits, node_logits = self.decode(z, batch_index, ends)

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