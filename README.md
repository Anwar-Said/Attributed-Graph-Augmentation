source code for the paper entitled "On Augmenting Topological Graph Representations for Attributed Graphs" <br>

Requirements: <br>
python3 <br>
networkx ==2.5 <br>
rdkit==2020.09.1 <br>
sklearn==1.0
grakel



Running model

This paper proposes a novel graph augmentation framework that transforms attributed graphs into unattributed graphs while preserving the attribute information. This allows directly applying graph kernels and other graph descriptors on the enhanced augmented graphs to improve their performance. The proposed framework was evaluated on toxicity prediction task, for which several molecular datasets were considered. Please see the paper for the dataset sources and formats. <br>

This repository provides the source code for the proposed approach and can be run in the following way. 

For easy reproducability of the results, we provide all the augmented datasets using the proposed approach in two directories: data_cliques and lollipop. Data_cliques direcotry contains the graphs where clique augmentation has been performed (Table 2 in the paper). Lollipop directory contains the augmented lollipop graphs respectively. We also provide the simple graphs where node attributes were ignored in "smiles" directory. 

we have applied different methods to compare the results on simple and augmented graphs. Here we provide the source codes for three methods: NetLSD, shortest path kernel and WL kernel. We provide the following files:


netlsd.py: this file contains the implementation for the netlsd method. It has three main components: first, it iterates on all the datasets for cliques augmented graphs and report the accuracy on every dataset. This experiment's results are reported in Table 2 under NetLSD (\mathcal{G}). 

Secondly, it iterates over all simple graphs (in smiles directory), run the same RF classifier and report the results. These results are reported in Table 2 under NetLSD (first column (G)).

Finally, it iterates over all augmented lollipop graphs and report the results presented in Table 3 under NetLSD (\mathcal{G}). 

Other files such as shortest path kernel.py and kernels.py do the same.



To regenerate the augmented graphs, we also provide the source code in main.py file along with datasets in smile format (smile directory). This source code can be used to generate the augmented graphs. 

Paper <br>

@article{said2023augmenting,<br>
  title={On augmenting topological graph representations for attributed graphs},<br>
  author={Said, Anwar and Shabbir, Mudassir and Hassan, Saeed-Ul and Hassan, Zohair Raza and Ahmed, Ammar and Koutsoukos, Xenofon},<br>
  journal={Applied Soft Computing},<br>
  pages={110104},<br>
  year={2023},<br>
  publisher={Elsevier}<br>
}<br>

