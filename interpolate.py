import sys
import torch
from gvae import GVAE
from config import DEVICE as device
from dataset import MoleculeDataset
from utils import slice_atom_type_from_node_feats
import deepchem as dc
import torch.nn.functional as F


model_path = sys.argv[1]

seeds = open("./data/raw/Seeds.csv", "r").readlines()

NUM_POINTS = 5
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)


model = torch.load(model_path)

MAX_MOLECULE_SIZE = model.max_num_atoms

def proces_smiles(smiles):
    f = featurizer.featurize(smiles)
    data = f[0].to_pyg_graph()
    data.y = 0
    data.smiles = smiles
    atom_types = slice_atom_type_from_node_feats(data.x)


    if (data.x.shape[0] < MAX_MOLECULE_SIZE) and -1 not in atom_types:
        # pad the molecule .node_feature matrix to have MAX_MOLECULE_SIZE rows
        existing = len(f[0].node_features)
        rows_needed = MAX_MOLECULE_SIZE - existing

        data.x  = F.pad(data.x, (0, 0, 0, rows_needed))

        return data
    else:
        return None

    

print("Starting interpolation")
data = [proces_smiles(seed) for seed in seeds if proces_smiles(seed) is not None]
model.interpolate(data, NUM_POINTS)

