from rdkit import Chem
import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# from rdkit.Chem import Draw
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import csv
import itertools
import scipy as sp
import time
from scipy import sparse
import scipy.io
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


def read_data(pathX, target):  
    with open(pathX) as f:
        content = f.readlines()
    X = [x.strip() for x in content] 

    l = np.load(target)
    labels = []
    for lab in l:
        if lab==1:
            labels.append(lab)
        else:
            labels.append(0)
    return X,labels
def construct_simpleGraphs(data, labels):
    graph_list = []
    lab = []
    for index,smile in enumerate(data):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        adj = Chem.GetAdjacencyMatrix(mol)
        g = nx.from_numpy_array(adj)
        graph_list.append(g)
        lab.append(labels[index])
    return graph_list, lab

def construct_augmented_graph(data, labels,k):
    graph_list = []
    lab = []
    max_ = 0
    min_ = 999999999
    for index,smile in enumerate(data):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        atoms = [atom for atom in mol.GetAtoms()]
        non_metals = ['H','He','C','N','O','F','Ne','P','S','Cl','Ar','Se','Br','Kr','I','Xe','At','Rn','Og']
        adj = Chem.GetAdjacencyMatrix(mol)
        g = nx.from_numpy_array(adj)
        for i, n in enumerate(atoms):
            nodes_cpy = list(g.nodes())
            if (n.GetIsAromatic()) and (n.GetFormalCharge() == 0):
                last_node = sorted(nodes_cpy)[-1]
                k = 4
                node_list = list(np.arange(last_node+1,last_node+k+1))
                g.add_nodes_from(node_list)
                edge_list = itertools.combinations(node_list, 2)
                g.add_edges_from(edge_list)
                g.add_edge(n.GetIdx(), node_list[0])

        for i, n in enumerate(atoms):
            nodes_cpy = list(g.nodes())
            if (n.GetFormalCharge() != 0):
                if (n.GetFormalCharge() == 1) and (n.GetIsAromatic() == False):
                    last_node = sorted(nodes_cpy)[-1]
                    k = 6
                    node_list = list(np.arange(last_node+1,last_node+k+1))
                    g.add_nodes_from(node_list)
                    edge_list = list(itertools.combinations(node_list, 2))
                    edge_list = edge_list[:-2]
                    g.add_edges_from(edge_list)
                    g.add_edge(n.GetIdx(), node_list[0])
            elif (n.GetFormalCharge() == -1):
                k = 7
                last_node = sorted(nodes_cpy)[-1]
                node_list = list(np.arange(last_node+1,last_node+k+1))
                g.add_nodes_from(node_list)
                edge_list = list(itertools.combinations(node_list, 2))
                edge_list = edge_list[:-4]
                g.add_edges_from(edge_list)
                g.add_edge(n.GetIdx(), node_list[0])
                nodes_cpy.extend(node_list)

        
        for i, n in enumerate(atoms):
            nodes_cpy = list(g.nodes())
            symbol = n.GetSymbol()

            if (symbol in non_metals) and (n.GetFormalCharge() == 0):
                k =8
                last_node = sorted(nodes_cpy)[-1]
                node_list = list(np.arange(last_node+1,last_node+k+1))
                g.add_nodes_from(node_list)
                edge_list = list(itertools.combinations(node_list, 2))
                edge_list = edge_list[:-6]
                g.add_edges_from(edge_list)
                g.add_edge(n.GetIdx(), node_list[0])

        #*******************************************ADDING EDGES INFO********************************************************888
        bonds = [x for x in mol.GetBonds()]
        for b in bonds:
            nodes_cpy = list(g.nodes())
            if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                k = 5
                first_node = sorted(nodes_cpy)[-1]+1
                nodes_list = np.arange(first_node,first_node+k)
                g.add_nodes_from(nodes_list)
                comb = list(itertools.combinations(nodes_list, 2))
                conn_edges = [(b.GetBeginAtom().GetIdx(),first_node), (b.GetEndAtom().GetIdx(),first_node)]
                comb.extend(conn_edges)
                g.add_edges_from(comb)
            nodes_cpy = list(g.nodes())
            if b.GetBondType() == Chem.rdchem.BondType.SINGLE:
                k = 9
                first_node = sorted(nodes_cpy)[-1]+1
                nodes_list = np.arange(first_node,first_node+(k+1))
                g.add_nodes_from(nodes_list)
                comb = list(itertools.combinations(nodes_list, 2))
                comb = comb[:-2]
                conn_edges = [(b.GetBeginAtom().GetIdx(),first_node), (b.GetEndAtom().GetIdx(),first_node)]
                comb.extend(conn_edges)
                g.add_edges_from(comb)
            nodes_cpy = list(g.nodes())
            if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                k = 10
                first_node = sorted(nodes_cpy)[-1]+1
                nodes_list = np.arange(first_node,first_node+(k+1))
                g.add_nodes_from(nodes_list)
                comb = list(itertools.combinations(nodes_list, 2))
                comb = comb[:-4]
                conn_edges = [(b.GetBeginAtom().GetIdx(),first_node), (b.GetEndAtom().GetIdx(),first_node)]
                comb.extend(conn_edges)
                g.add_edges_from(comb)
            if b.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                k = 11
                first_node = sorted(nodes_cpy)[-1]+1
                nodes_list = np.arange(first_node,first_node+(k+1))
                g.add_nodes_from(nodes_list)
                comb = list(itertools.combinations(nodes_list, 2))
                comb = comb[:-6]
                conn_edges = [(b.GetBeginAtom().GetIdx(),first_node), (b.GetEndAtom().GetIdx(),first_node)]
                comb.extend(conn_edges)
                g.add_edges_from(comb)
        if g.number_of_nodes()>max_:
            max_=g.number_of_nodes()
        if g.number_of_nodes()<min_:
            min_ = g.number_of_nodes()
        graph_list.append(g)
        lab.append(labels[index])
    return graph_list, lab,max_,min_

### read the smiles from the directory and construct augmented and original graphs

path = 'data/smiles/'
dataset = 'nr-ahr_sm.can'
target = 'nr-ahr_y.npy'
data,y = read_data(path+dataset,path+target)
print("data loaded successfully with {} number of graphs and {} labels".format(len(data),len(y)))
dir_save = "data/augmented_graphs/"
np.save(dir_save+dataset+"_g.npy",data)
np.save(dir_save+dataset+"_y.npy",y)
print("augmented graphs has been saved successfully!")


### construct simple graphs

path = 'data/smiles/'
dataset = 'nr-ahr_sm.can'
target = 'nr-ahr_y.npy'
data,y = construct_simpleGraphs(path+dataset,path+target)
print("data loaded successfully with {} number of graphs and {} labels".format(len(data),len(y)))
dir_save = "data/simple_graphs/"
np.save(dir_save+dataset+"_g.npy",data)
np.save(dir_save+dataset+"_y.npy",y)
print("graphs has been saved successfully!")



