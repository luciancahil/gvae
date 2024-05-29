import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
from tqdm import tqdm
import deepchem as dc
from config import MAX_MOLECULE_SIZE
from utils import slice_atom_type_from_node_feats
import re
import torch.nn.functional as F

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, length=0):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.elements = []
        self.length = length
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """

        # always process
        return "Never"
        

    def download(self):
        pass

    def process(self):
        f = open(self.raw_paths[0], 'r')
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for line in f:
            # Featurize molecule
            index += 1
            f = featurizer.featurize(line)
            data = f[0].to_pyg_graph()
            data.y = 0
            data.smiles = line

            # Get the molecule's atom types
            atom_types = slice_atom_type_from_node_feats(data.x)

            # Only save if molecule is in permitted size
            if (data.x.shape[0] < MAX_MOLECULE_SIZE) and -1 not in atom_types:
                # pad the molecule .node_feature matrix to have MAX_MOLECULE_SIZE rows
                existing = len(f[0].node_features)
                rows_needed = MAX_MOLECULE_SIZE - existing

                data.x  = F.pad(data.x, (0, 0, 0, rows_needed))

                self.elements.append(data)
                self.length += 1
            else:
                pass
                #print("Skipping invalid mol (too big/unknown atoms): ", data.smiles)
        print(f"Done. Stored {self.length} preprocessed molecules.")

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        return self.elements[idx]