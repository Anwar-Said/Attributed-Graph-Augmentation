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
    min_ = 999999999999999999999999
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

def apply_RF(feature_matrix, labels):
    model = RandomForestClassifier(n_estimators=500)
    acc = []
    for i in range(5):
        res = cross_val_score(model, feature_matrix, labels, cv=10, scoring='accuracy')
        acc.append(np.mean(res))
    return np.mean(acc)


# In[ ]:


"""Shortest-Path graph kernel.
Python implementation based on: "Shortest-path kernels on graphs", by
Borgwardt, K.M.; Kriegel, H.-P., in Data Mining, Fifth IEEE
International Conference on , vol., no., pp.8 pp.-, 27-30 Nov. 2005
doi: 10.1109/ICDM.2005.132
Author : Sandro Vega-Pons, Emanuele Olivetti
"""


class Shortest_Path_Kernel:
    """
    Shorthest path graph kernel.
    """
    def get_singature(self,data, verbose = False):
        max_len = 0
        hist  =[]
        for g in data:
            if isinstance(g,np.ndarray):
                g = nx.from_numpy_array(g)
#             g = g.astype('float32')    
            fwm1 = np.array(nx.floyd_warshall_numpy(g))
            fwm1 = np.where(fwm1 == np.inf, 0, fwm1)
            fwm1 = np.where(fwm1 == np.nan, 0, fwm1)
            fwm1 = np.triu(fwm1, k=1)
            bc = np.bincount(fwm1.reshape(-1).astype(int))
            if len(bc)>max_len:
                max_len = len(bc)
            hist.append(bc)
        feature_matrix = []
        for h in hist:
            v = np.zeros((max_len,),dtype = int)
            v[range(0, len(h)-1)] = h[1:]
            feature_matrix.append(v)
        return feature_matrix

        
    def compare(self, g_1, g_2, verbose=False):
        """Compute the kernel value (similarity) between two graphs.
        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.
        Returns
        -------
        k : The similarity value between g1 and g2.
        """
        # Diagonal superior matrix of the floyd warshall shortest
        # paths:
        fwm1 = np.array(nx.floyd_warshall_numpy(g_1))
        fwm1 = np.where(fwm1 == np.inf, 0, fwm1)
        fwm1 = np.where(fwm1 == np.nan, 0, fwm1)
        fwm1 = np.triu(fwm1, k=1)
        bc1 = np.bincount(fwm1.reshape(-1).astype(int))

        fwm2 = np.array(nx.floyd_warshall_numpy(g_2))
        fwm2 = np.where(fwm2 == np.inf, 0, fwm2)
        fwm2 = np.where(fwm2 == np.nan, 0, fwm2)
        fwm2 = np.triu(fwm2, k=1)
        bc2 = np.bincount(fwm2.reshape(-1).astype(int))

        # Copy into arrays with the same length the non-zero shortests
        # paths:
        v1 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v1[range(0, len(bc1)-1)] = bc1[1:]

        v2 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v2[range(0, len(bc2)-1)] = bc2[1:]

        return np.sum(v1 * v2)

    def compare_normalized(self, g_1, g_2, verbose=False):
        """Compute the normalized kernel value between two graphs.
        A normalized version of the kernel is given by the equation:
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2))
        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.
        Returns
        -------
        k : The similarity value between g1 and g2.
        """
        return self.compare(g_1, g_2) / (np.sqrt(self.compare(g_1, g_1) *
                                                 self.compare(g_2, g_2)))

    def compare_list(self, graph_list, verbose=False):
        """Compute the all-pairs kernel values for a list of graphs.
        This function can be used to directly compute the kernel
        matrix for a list of graphs. The direct computation of the
        kernel matrix is faster than the computation of all individual
        pairwise kernel values.
        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)
        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.
        """
        n = len(graph_list)
        k = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k[i, j] = self.compare(graph_list[i], graph_list[j])
                k[j, i] = k[i, j]

        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

        return k_norm


# In[ ]:


path_read = 'data/augmented data/data_cliques/'
files = ['mutag','nr-ahr']
obj = Shortest_Path_Kernel()
for i in range(len(files)):
    graphs = np.load(path_read+files[i]+'_g.npy', allow_pickle = True)
    y = np.load(path_read+files[i]+'_y.npy', allow_pickle = True)
    print("data loaded:",files[i], len(graphs),len(y),type(graphs[0]))
    feature_matrix_sp = obj.get_singature(graphs)
    print("sp features constructed")
    algo = 'sp'
    acc = apply_RF(feature_matrix_sp,y)
    print("accuracy:",acc)
    
