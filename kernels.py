#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
print(__doc__)
import numpy as np
# import matplotlib.pyplot as plt
from grakel import GraphKernel, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import csv
from time import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import igraph
import networkx as nx




def convert_graph(data):
    graph_list = []
    node_index = 0
    for g in data:
        if isinstance(g, np.ndarray):
            g = nx.from_numpy_array(g)
        g = nx.convert_node_labels_to_integers(g, first_label = node_index)
        node_index = max(list(g.nodes()))+1
        edges = set(list(g.edges()))
        labels = {}
        for n in g.nodes():
            labels[n] = 0
        edges_labels = {}
        for e in g.edges():
            edges_labels[e] = 0
        g_list = [edges,labels,edges_labels]
        graph_list.append(g_list)
    return graph_list



### WL on original datasets
files = ['mutag','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("kernels.csv",'a',newline = '')
path = 'data/smiles/'
res_writer = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["dataset","algorithm","accuracy original"])
for d in files:
    graphs = np.load(path+d+'_g.npy', allow_pickle = True)
    y = np.load(path+d+'_y.npy', allow_pickle = True)
    print("data loaded:",d, len(graphs),len(y))
    G = convert_graph(graphs)
    gk = GraphKernel(kernel={"name": "WL"})
    rf = RandomForestClassifier(n_estimators = 500)
    estimator = make_pipeline(gk,rf)
    res = cross_val_score(estimator, G,y, cv=10, scoring='accuracy',error_score='raise')
    acc = np.mean(res)
    algo = 'NHK'
    to_write = [d,algo,acc]
    print(to_write)
    res_writer.writerow(to_write)
    file.flush()
file.close()
    


### running on augmented clique graphs

files = ['mutag','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("wl.csv",'a',newline = '')
path = 'data/data_cliques/'
res_writer = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["dataset","algorithm","accuracy original"])
for d in files:
    graphs = np.load(path+d+'_g.npy', allow_pickle = True)
    y = np.load(path+d+'_y.npy', allow_pickle = True)
    print("data loaded:",d, len(graphs),len(y))
    G = convert_graph(graphs)
    gk = GraphKernel(kernel={"name": "WL"})
    rf = RandomForestClassifier(n_estimators = 500)
    estimator = make_pipeline(gk,rf)
    res = cross_val_score(estimator, G,y, cv=10, scoring='accuracy',error_score='raise')
    acc = np.mean(res)
    to_write = [d,acc]
    print(to_write)
    res_writer.writerow(to_write)
    file.flush()
file.close()

### running on augmented lollipop graphs

files = ['mutag','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("wl.csv",'a',newline = '')
path = 'data/lollipop/'
res_writer = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["dataset","algorithm","accuracy original"])
for d in files:
    graphs = np.load(path+d+'_g.npy', allow_pickle = True)
    y = np.load(path+d+'_y.npy', allow_pickle = True)
    print("data loaded:",d, len(graphs),len(y))
    G = convert_graph(graphs)
    gk = GraphKernel(kernel={"name": "WL"})
    rf = RandomForestClassifier(n_estimators = 500)
    estimator = make_pipeline(gk,rf)
    res = cross_val_score(estimator, G,y, cv=10, scoring='accuracy',error_score='raise')
    acc = np.mean(res)
    to_write = [d,acc]
    print(to_write)
    res_writer.writerow(to_write)
    file.flush()
file.close()