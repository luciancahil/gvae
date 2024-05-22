Graph Variational Autoencoder with Pytorch Geometric for Molecule Generation.

This model is based on this paper:
https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-019-0396-x.pdf

Note that the model is not fully functioning yet.

Please have a look at the original implementation in tensorflow for further directions:
https://github.com/seokhokang/graphvae_approx/


set up conda:

`````
conda create -n gvae
conda activate gvae
conda install -q -y pyg -c pyg
conda install -q -y pytorch cudatoolkit=11.3 -c pytorch
pip install pandas
pip install deepchem
pip install mlflow

`````