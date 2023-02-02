#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
# import sys,igraph
import pandas as pd
import networkx as nx
import scipy.io
from sklearn.utils import shuffle
# from grakel import GraphKernel, datasets
import csv
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


def check_1d(inp):
    """
    Check input to be a vector. Converts lists to np.ndarray.

    Parameters
    ----------
    inp : obj
        Input vector

    Returns
    -------
    numpy.ndarray or None
        Input vector or None

    Examples
    --------
    >>> check_1d([0, 1, 2, 3])
    [0, 1, 2, 3]

    >>> check_1d('test')
    None

    """
    if isinstance(inp, list):
        return check_1d(np.array(inp))
    if isinstance(inp, np.ndarray):
        if inp.ndim == 1: # input is a vector
            return inp


def check_2d(inp):
    """
    Check input to be a matrix. Converts lists of lists to np.ndarray.

    Also allows the input to be a scipy sparse matrix.
    
    Parameters
    ----------
    inp : obj
        Input matrix

    Returns
    -------
    numpy.ndarray, scipy.sparse or None
        Input matrix or None

    Examples
    --------
    >>> check_2d([[0, 1], [2, 3]])
    [[0, 1], [2, 3]]

    >>> check_2d('test')
    None

    """
    if isinstance(inp, list):
        return check_2d(np.array(inp))
    if isinstance(inp, (np.ndarray, np.matrixlib.defmatrix.matrix)):
        if inp.ndim == 2: # input is a dense matrix
            return inp
    if sps.issparse(inp):
        if inp.ndim == 2: # input is a sparse matrix
            return inp


def graph_to_laplacian(G, normalized=True):
    """
    Converts a graph from popular Python packages to Laplacian representation.

    Currently support NetworkX, graph_tool and igraph.
    
    Parameters
    ----------
    G : obj
        Input graph
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.

    Returns
    -------
    scipy.sparse
        Laplacian matrix of the input graph

    Examples
    --------
    >>> graph_to_laplacian(nx.complete_graph(3), 'unnormalized').todense()
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]

    >>> graph_to_laplacian('test')
    None

    """
    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            if normalized:
                return nx.normalized_laplacian_matrix(G)
            else:
                return nx.laplacian_matrix(G)
    except ImportError:
        pass
    try:
        import graph_tool.all as gt
        if isinstance(G, gt.Graph):
            if normalized:
                return gt.laplacian_type(G, normalized=True)
            else:
                return gt.laplacian(G)
    except ImportError:
        pass
    try:
        import igraph as ig
        if isinstance(G, ig.Graph):
            if normalized:
                return np.array(G.laplacian(normalized=True))
            else:
                return np.array(G.laplacian())
    except ImportError:
        pass


def mat_to_laplacian(mat, normalized):
    """
    Converts a sparse or dence adjacency matrix to Laplacian.
    
    Parameters
    ----------
    mat : obj
        Input adjacency matrix. If it is a Laplacian matrix already, return it.
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.

    Returns
    -------
    obj
        Laplacian of the input adjacency matrix

    Examples
    --------
    >>> mat_to_laplacian(numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), False)
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]

    """
    if sps.issparse(mat):
        if np.all(mat.diagonal()>=0): # Check diagonal
            if np.all((mat-sps.diags(mat.diagonal())).data <= 0): # Check off-diagonal elements
                return mat
    else:
        if np.all(np.diag(mat)>=0): # Check diagonal
            if np.all(mat - np.diag(mat) <= 0): # Check off-diagonal elements
                return mat
    deg = np.squeeze(np.asarray(mat.sum(axis=1)))
    if sps.issparse(mat):
        L = sps.diags(deg) - mat
    else:
        L = np.diag(deg) - mat
    if not normalized:
        return L
    with np.errstate(divide='ignore'):
        sqrt_deg = 1.0 / np.sqrt(deg)
    sqrt_deg[sqrt_deg==np.inf] = 0
    if sps.issparse(mat):
        sqrt_deg_mat = sps.diags(sqrt_deg)
    else:
        sqrt_deg_mat = np.diag(sqrt_deg)
    return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


def updown_linear_approx(eigvals_lower, eigvals_upper, nv):
    """
    Approximates Laplacian spectrum using upper and lower parts of the eigenspectrum.
    
    Parameters
    ----------
    eigvals_lower : numpy.ndarray
        Lower part of the spectrum, sorted
    eigvals_upper : numpy.ndarray
        Upper part of the spectrum, sorted
    nv : int
        Total number of nodes (eigenvalues) in the graph.

    Returns
    -------
    numpy.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> updown_linear_approx([1, 2, 3], [7, 8, 9], 9)
    array([1,  2,  3,  4,  5,  6,  7,  8,  9])

    """
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError('Number of supplied eigenvalues ({0} lower and {1} upper) is higher than number of nodes ({2})!'.format(nal, nau, nv))
    ret = np.zeros(nv)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    ret[nal-1:-nau+1] = np.linspace(eigvals_lower[-1], eigvals_upper[0], nv-nal-nau+2)
    return ret


def eigenvalues_auto(mat, n_eivals='auto'):
    """
    Automatically computes the spectrum of a given Laplacian matrix.
    
    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse
        Laplacian matrix
    n_eivals : string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.

    Returns
    -------
    np.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> eigenvalues_auto(numpy.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]), 'auto')
    array([0, 3, 3])

    """
    do_full = True
    n_lower = 150
    n_upper = 150
    nv = mat.shape[0]
    if n_eivals == 'auto':
        if mat.shape[0] > 1024:
            do_full = False
    if n_eivals == 'full':
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        do_full = False
    if do_full and sps.issparse(mat):
        mat = mat.todense()
    if sps.issparse(mat):
        if n_lower == n_upper:
            tr_eivals = spsl.eigsh(mat, 2*n_lower, which='BE', return_eigenvectors=False)
            return updown_linear_approx(tr_eivals[:n_upper], tr_eivals[n_upper:], nv)
        else:
            lo_eivals = spsl.eigsh(mat, n_lower, which='SM', return_eigenvectors=False)[::-1]
            up_eivals = spsl.eigsh(mat, n_upper, which='LM', return_eigenvectors=False)
            return updown_linear_approx(lo_eivals, up_eivals, nv)
    else:
        if do_full:
            return spl.eigvalsh(mat)
        else:
            lo_eivals = spl.eigvalsh(mat, eigvals=(0, n_lower-1))
            up_eivals = spl.eigvalsh(mat, eigvals=(nv-n_upper-1, nv-1))
            return updown_linear_approx(lo_eivals, up_eivals, nv)


# In[ ]:


def compare(descriptor1, descriptor2):
    """
    Computes the distance between two NetLSD representations.
    
    Parameters
    ----------
    descriptor1: numpy.ndarray
        First signature to compare
    descriptor2: numpy.ndarray
        Second signature to compare

    Returns
    -------
    float
        NetLSD distance

    """
    return np.linalg.norm(descriptor1-descriptor2)


def netlsd(inp, timescales=np.logspace(-2, 2, 250), kernel='heat', eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes NetLSD signature from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    kernel : str
        Either 'heat' or 'wave'. Type of a kernel to use for computation.
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        NetLSD signature

    """
    if kernel not in {'heat', 'wave'}:
        raise AttributeError('Unirecognized kernel type: expected one of [\'heat\', \'wave\'], got {0}'.format(kernel))
    if not isinstance(normalized_laplacian, bool):
        raise AttributeError('Unknown Laplacian type: expected bool, got {0}'.format(normalized_laplacian))
    if not isinstance(eigenvalues, (int, tuple, str)):
        raise AttributeError('Unirecognized requested eigenvalue number: expected type of [\'str\', \'tuple\', or \'int\'], got {0}'.format(type(eigenvalues)))
    if not isinstance(timescales, np.ndarray):
        raise AttributeError('Unirecognized timescales data type: expected np.ndarray, got {0}'.format(type(timescales)))
    if timescales.ndim != 1:
        raise AttributeError('Unirecognized timescales dimensionality: expected a vector, got {0}-d array'.format(timescales.ndim))
    if normalization not in {'complete', 'empty', 'none', True, False, None}:
        if not isinstance(normalization, np.ndarray):
            raise AttributeError('Unirecognized normalization type: expected one of [\'complete\', \'empty\', None or np.ndarray], got {0}'.format(normalization))
        if normalization.ndim != 1:
            raise AttributeError('Unirecognized normalization dimensionality: expected a vector, got {0}-d array'.format(normalization.ndim))
        if timescales.shape[0] != normalization.shape[0]:
            raise AttributeError('Unirecognized normalization dimensionality: expected {0}-length vector, got length {1}'.format(timescales.shape[0], normalization.shape[0]))

    eivals = check_1d(inp)
    if eivals is None:
        mat = check_2d(inp)
        if mat is None:
            mat = graph_to_laplacian(inp, normalized_laplacian)
            if mat is None:
                raise ValueError('Unirecognized input type: expected one of [\'np.ndarray\', \'scipy.sparse\', \'networkx.Graph\',\' graph_tool.Graph,\' or \'igraph.Graph\'], got {0}'.format(type(inp)))
        else:
            mat = mat_to_laplacian(inp, normalized_laplacian)
        eivals = eigenvalues_auto(mat, eigenvalues)
    if kernel == 'heat':
        return _hkt(eivals, timescales, normalization, normalized_laplacian)
    else:
        return _wkt(eivals, timescales, normalization, normalized_laplacian)


def heat(inp,scale, eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes heat kernel trace from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature

    """
    timescales = np.logspace(-2, 2, scale)
    return netlsd(inp, timescales, 'heat', eigenvalues, normalization, normalized_laplacian)


def wave(inp, scale, eigenvalues='auto', normalization='empty', normalized_laplacian=True):
    """
    Computes wave kernel trace from some given input, timescales, and normalization.

    Accepts matrices, common Python graph libraries' graphs, or vectors of eigenvalues. 
    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    inp: obj
        2D numpy/scipy matrix, common Python graph libraries' graph, or vector of eigenvalues
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    eigenvalues : str
        Either string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized wave kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Wave kernel trace signature

    """
    timescales = np.linspace(0, 2 * np.pi, scale)
    return netlsd(inp, timescales, 'wave', eigenvalues, normalization, normalized_laplacian)


def _hkt(eivals, timescales, normalization, normalized_laplacian):
    """
    Computes heat kernel trace from given eigenvalues, timescales, and normalization.

    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    eivals : numpy.ndarray
        Eigenvalue vector
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature

    """
    nv = eivals.shape[0]
    hkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        # print("idx {} and t {} : ".format(idx, t))
        hkt[idx] = np.sum(np.exp(-t * eivals))
    if isinstance(normalization, np.ndarray):
        return hkt / normalization
    if normalization == 'empty' or normalization == True:
        return hkt / nv
    if normalization == 'complete':
        if normalized_laplacian:
            return hkt / (1 + (nv - 1) * np.exp(-timescales))
        else:
            return hkt / (1 + nv * np.exp(-nv * timescales))
    return hkt


def _wkt(eivals, timescales, normalization, normalized_laplacian):
    """
    Computes wave kernel trace from given eigenvalues, timescales, and normalization.

    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    eivals : numpy.ndarray
        Eigenvalue vector
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized wave kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Wave kernel trace signature

    """
    nv = eivals.shape[0]
    wkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        wkt[idx] = np.sum(np.exp(-1j * t * eivals))
    if isinstance(normalization, np.ndarray):
        return wkt / normalization
    if normalization == 'empty' or normalization == True:
        return wkt / nv
    if normalization == 'complete':
        if normalized_laplacian:
            return wkt / (1 + (nv - 1) * np.cos(timescales))
        else:
            return wkt / (1 + (nv - 1) * np.cos(nv * timescales))
    return wkt


# In[ ]:


#fetches dataset from graph kernel website
def return_dataset(file_name):
    #i = 'G_nci1'
    dd = datasets.fetch_dataset(file_name,verbose = True)
    graph_list = []
    node_attr = []
    for gg in dd.data:
        v = set([i[0] for i in gg[0]]).union(set([i[1] for i in gg[0]]))
        g_ = igraph.Graph()
        g_.add_vertices([str(i) for i in v])
        g_.add_edges([(str(i[0]), str(i[1])) for i in gg[0]])
        g_.simplify()
        A = g_.get_edgelist()
        g = nx.Graph(A)
        graph_list.append(g)
        g_.vs['idx'] = [str(i) for i in g_.vs.indices]
        node_attr.append(gg[1])
    data_y = dd.target
    return graph_list, data_y, node_attr
def apply_netlsdwave(graphs,scale):
    feature_matrix = []
    for g in graphs:
        feature_matrix.append(wave(g,scale))
    return np.array(feature_matrix)
def apply_netlsdheat(graphs,scale):
    feature_matrix = []
    for g in graphs:
        feature_matrix.append(heat(g,scale))
    return np.array(feature_matrix)
def apply_RF(feature_matrix, labels):
    model = RandomForestClassifier(n_estimators=500)
    res = cross_val_score(model, feature_matrix, labels, cv=10, scoring='accuracy')
    return np.mean(res)


# In[ ]:


### Applying on the augmented cliques data


# In[ ]:


data = ['mutag','PTC','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
scale = 250
file = open("netlsd_cliques.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["dataset","accuracy","cliques"])
path = "data/data_cliques/"
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("data loaded:",d, len(data),len(labels))
    accuracy = []
    for i in range(10):
        ## netlsd involves approximation, which may results a slight variance in the results
        feature_matrix = apply_netlsdheat(data,scale)
        acc = apply_RF(feature_matrix, labels)
        accuracy.append(acc)
    to_write = [d,np.mean(accuracy)]
    res_writer.writerow(to_write)
    file.flush()
    print("complete processing dataset ",d, "accuracy:", np.mean(accuracy))
file.close()    


# In[ ]:
### results on simple graphs without augmentation

path = "data/smiles/"
data = ['mutag','PTC','NCI1','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
file = open("netlsd.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("{} loaded".format(d))
    accuracy = []
    for i in range(10): ## iterating 10 times to get stable results
        ## netlsd involves approximation, which may results a slight variance in the results
        feature_matrix = apply_netlsdheat(data,scale)
        acc = apply_RF(feature_matrix, labels)
        accuracy.append(acc)
    to_write = [d,np.mean(accuracy)]
    res_writer.writerow(to_write)
    file.flush()
    print("complete processing dataset ",d, "accuracy:", np.mean(accuracy))
#         break
file.close()



## reproducing results on lollipop augmented graphs


# In[ ]:


data = ['mutag','PTC','nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
scale = 250
file = open("netlsd_lollipop.csv",'a',newline = '')
res_writer = csv.writer(file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
res_writer.writerow(["dataset","accuracy","lollipop"])
path = "data/lollipop/"
for d in data:
    data = np.load(path+d+'_g.npy',allow_pickle = True)
    labels = np.load(path+d+'_y.npy',allow_pickle = True)
    print("data loaded:",d, len(data),len(labels))
    accuracy = []
    for i in range(5):
        ## netlsd involves approximation, which may results a slight variance in the results
        feature_matrix = apply_netlsdheat(data,scale)
        acc = apply_RF(feature_matrix, labels)
        accuracy.append(acc)
    to_write = [d,np.mean(accuracy)]
    res_writer.writerow(to_write)
    file.flush()
    print("complete processing dataset ",d, "accuracy:", np.mean(accuracy))
file.close()    


