#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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


def apply_RF(feature_matrix, labels):
    model = RandomForestClassifier(n_estimators=500)
    res = cross_val_score(model, feature_matrix, labels, cv=10, scoring='accuracy')
    return np.mean(res)



### applying on original graphs --- no augmentation
obj = Shortest_Path_Kernel()
path = "data/smiles/"
data = ['mutag','PTC','NCI1','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("SP.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("{} loaded".format(d))
    accuracy = []
    for i in range(10):
        feature_matrix = obj.get_singature(data)
        acc = apply_RF(feature_matrix,labels)
        accuracy.append(acc)
    print("accuracy {} on dataset: {}".format(np.mean(accuracy), d)) 
    to_write = [d,np.mean(acc)]
    res_writer.writerow(to_write)
    file.flush()
#         break
file.close()



obj = Shortest_Path_Kernel()
path = "data/data_cliques/"
data = ['mutag','PTC','NCI1','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("SP.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("{} loaded".format(d))
    accuracy = []
    for i in range(10):
        feature_matrix = obj.get_singature(data)
        acc = apply_RF(feature_matrix,labels)
        accuracy.append(acc)
    print("accuracy {} on dataset: {}".format(np.mean(accuracy), d)) 
    to_write = [d,np.mean(acc)]
    res_writer.writerow(to_write)
    file.flush()
#         break
file.close()



### applying on lollipop graphs
obj = Shortest_Path_Kernel()
path = "data/lollipop/"
data = ['mutag','PTC','NCI1','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("SP.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["SP results on large cliques data"])
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("{} loaded".format(d))
    accuracy = []
    for i in range(10):
        feature_matrix = obj.get_singature(data)
        acc = apply_RF(feature_matrix,labels)
        accuracy.append(acc)
    print("accuracy {} on dataset: {}".format(np.mean(accuracy), d)) 
    to_write = [d,np.mean(acc)]
    res_writer.writerow(to_write)
    file.flush()
    break
file.close()




