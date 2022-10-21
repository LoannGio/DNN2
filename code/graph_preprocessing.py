from spektral.utils.convolution import chebyshev_filter, gcn_filter
from scipy.sparse.csgraph import floyd_warshall
from pyspark import SparkContext, SparkConf
import numpy as np
import random
import networkx as nx
import time
from scipy.sparse import csr_matrix
from scipy.io import mmread
from tulip import tlp

def AM2Chebyshev(AM, K):
    return chebyshev_filter(AM, K) #returns matrix of shape (K+1, N, N)

def AM2GCNfilter(A, K): # K is here so that methods have same signature
    support = gcn_filter(A)
    return support.reshape(1, *support.shape) #return (1, N, N)
    
def graph2AM(g, increased=-1):
    AM_size = max(g.numberOfNodes(), int(increased))
    AM = np.zeros((AM_size, AM_size)) 
    for n in g.getNodes():
        for neighbor in g.getInOutNodes(n):
            AM[n.id][neighbor.id] = 1
    return AM

def AM2DM(AM, fill_value=0):
    DM = floyd_warshall(AM)
    if(type(fill_value) == int):
        DM[DM == np.inf] = fill_value
    elif(fill_value == "max"):
        tmp = -1
        DM[DM == np.inf] = tmp
        DM[DM == tmp] = np.max(DM)
    return DM

def graphFeatures(g, features_names, increased=-1):
    n_nodes = max(g.numberOfNodes(), increased)
    fill_value = np.nan
    n_features = len(features_names)
    for feature in features_names:
        if(feature.startswith("LAYOUT:")):
            n_features += 1
    F = np.full((n_nodes, n_features), fill_value)

    for feature in features_names:
        if(feature.startswith("LAYOUT:")):
            layout_algo = feature.split(":")[-1]
            applyTlpLayoutAlgorithm(g, layout_algo, "res_"+layout_algo) # will store the result layout in a property of the graph

    for n in g.getNodes():
        f = []
        for feature in features_names:
            if(feature.lower() == "id"):
                f.append(n.id)
            elif("random:" in feature):
                f.append(random.uniform(0, 1))
            elif("LAYOUT:" in feature):
                layout_algo = feature.split(":")[-1]
                pos3d = g.getProperty("res_"+layout_algo).getNodeValue(n)
                f.append(pos3d[0]) # X coord
                f.append(pos3d[1]) # Y coord            
        F[n.id] = f
    for i in range(F.shape[1]):
        v = F[:, i]
        v[np.isnan(v)] = np.min(v[~np.isnan(v)])
    return F

    
def scaleFeatures(F, scalers):
    n_features = F.shape[-1]
    assert len(scalers) == n_features
    for f in range(n_features):
        fs = F[:,f].astype(float)
        fs = fs.flatten()
        fs = fs.reshape((*fs.shape, 1))            
        fs = scalers[f].transform(fs)
        F[:,f] = fs.reshape(F[:,f].shape)
    return F


def numpy_findSigma(DM, real_n, perplexity=30, n_iters=20, dtype="float32", epsilon=1e-16):
    DM = DM.astype(dtype)
    DM_red = DM[:real_n, :real_n]
    X = DM_red
    sigma = np.ones((real_n))
    target = np.log(perplexity)
    P = np.maximum(local_pij_cond_var(X, sigma), epsilon)
    entropy = -np.sum(P * np.log(P), axis=1)

    sigmin = np.full((real_n), epsilon, dtype=dtype)
    sigmax = np.full((real_n), np.inf, dtype=dtype)

    upmin = np.where(entropy < target, sigma, sigmin)
    upmax = np.where(entropy > target, sigma, sigmax)

    for i in range(n_iters):
        P = np.maximum(local_pij_cond_var(X, sigma), epsilon)
        entropy = -np.sum(P * np.log(P), axis=1)
        if np.any(np.isnan(np.exp(entropy))):
            #recurs sigma, perplexity *= 1.5
            return numpy_findSigma(DM, real_n, perplexity=perplexity*1.5, n_iters=n_iters, dtype=dtype, epsilon=epsilon)
        upmin = np.where(entropy < target, sigma, sigmin)
        upmax = np.where(entropy > target, sigma, sigmax)
        sigmin = upmin
        sigmax = upmax

        upsigma = np.where(np.isinf(sigmax), sigma*2, (sigmin + sigmax) / 2.)
        sigma = upsigma
    return sigma

def local_pij_cond_var(X, sigma):
    N = X.shape[0]
    sqdistance = X**2
    esqdistance = np.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    np.fill_diagonal(esqdistance, 0)
    esqdistance_zd = esqdistance
    row_sum = np.sum(esqdistance_zd, axis=1).reshape((N, 1))
    return esqdistance_zd / row_sum  # Possibly dangerous

def getSigma(DM, real_n, perplexity=None, sigma_iters=20):
    N = DM.shape[-1]
    if(perplexity is None):
        (Pmin, Pmax) = (5,N/2)
        (Nmin, Nmax) = (2, N)
        # P = lambda n: (n-Nmin)*(Pmax-Pmin)/(Nmax-Nmin)+Pmin
        P = lambda n: (n-Nmin)*(Pmax-Pmin)/(Nmax-Nmin)+Pmin if Nmax>Nmin else Pmin
        #P = lambda n: (n-2)*45/126+5
        perplexity = P(real_n)
    sigma = numpy_findSigma(DM, real_n, perplexity=perplexity, n_iters=sigma_iters)
    
    filled_sigma = np.full((N), np.sum(sigma))
    filled_sigma[:real_n] = sigma    
    filled_sigma = filled_sigma.reshape((N, 1))
    return filled_sigma 

def swapMatrix(M, real_n=None, permutation=None):
    assert len(M.shape) == 2 and M.shape[0] == M.shape[1]
    N = M.shape[0]
    if(permutation is not None):
        assert len(permutation) == N
    else:
        permutation = np.random.permutation(N)

    mask = np.zeros((N,N))
    if(real_n is not None):
        assert real_n <= N
        mask[:real_n, :real_n] = 1
    else:
        mask[:,:] = 1
       
    swappedM = np.zeros((N, N), np.float32)
    swappedMask = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(N):
            swappedM[permutation[i]][permutation[j]] = M[i][j]
            swappedMask[permutation[i]][permutation[j]] = mask[i][j]
    return swappedM, swappedMask, permutation

def swapVector(V, real_n=None, permutation=None):
    assert len(V.shape) == 2
    N = V.shape[0]
    F = V.shape[1]
    if(permutation is not None):
        assert len(permutation) == N, str(len(permutation))+" vs. "+str(N)
    else:
        permutation = np.random.permutation(N)

    mask = np.zeros((N,1))
    if(real_n is not None):
        assert real_n <= N
        mask[:real_n] = 1
    else:
        mask[:] = 1
       
    swappedV = np.zeros(V.shape)
    swappedMask = np.zeros(mask.shape)
    for i in range(N):
        swappedV[permutation[i]] = V[i]
        swappedMask[permutation[i]] = mask[i]
    return swappedV, swappedMask, permutation


def AM2nx(AM):
    G = nx.Graph()
    N = AM.shape[0]
    for i in range(N):
        for j in range(i+1, N, 1):
            if(AM[i][j] == 1):
                G.add_edge(i,j)
    return G    

def nxPos2array(nxPos, N_max):
    pos_a = np.zeros((N_max, 2))
    for i in range(N_max):
        if(i in nxPos.keys()):
            pos_a[i] = nxPos[i]
    return pos_a
def AM2tlp(AM, mask):
    g = tlp.newGraph()
    N = AM.shape[0]
    real_n = int(np.sum(mask))
    nodes = g.addNodes(real_n)
    real_i, real_j = 0, 0
    id_mapping = {}
    for i in range(N):
        if(mask[i] == 1):
            id_mapping[real_i] = i
            for j in range(i, N, 1):
                if(AM[i][j] == 1):
                    g.addEdge(nodes[real_i], nodes[real_j])
                if(mask[j] == 1):
                    real_j += 1
            real_i += 1
            real_j = real_i
    return g, id_mapping

def pos2DM(pos2d, mask):
    diff = np.expand_dims(pos2d, 0) - np.expand_dims(pos2d, 1)
    squared_dist = np.sum(np.square(diff), axis=-1)
    DM = np.sqrt(squared_dist)
    matrice_mask = mask*mask.T
    DM*= matrice_mask
    return DM

def DM2edgesIdx(DM):
    return np.nonzero(DM == 1)

def shrinkDM(DM, mask):
    real_n = int(np.sum(mask))
    res = np.zeros((real_n, real_n))
    real_i, real_j = 0,0
    for i in range(DM.shape[0]):
        if(mask[i] == 1):
            for j in range(DM.shape[1]):
                if(mask[j] == 1):
                    res[real_i][real_j] = DM[i][j]
                    real_j +=1
            real_i +=1
            real_j = 0
    return res

def shrinkPos(pos, mask):
    real_n = int(np.sum(mask))
    res = np.zeros((real_n, 2))
    real_i = 0
    for i in range(pos.shape[0]):
        if(mask[i] == 1):
            res[real_i] = pos[i]
            real_i += 1
    return res

def nx2tlp(nxG, N_max):
    n = nxG.number_of_nodes()
    AM = np.zeros((N_max, N_max))
    AM[:n, :n] = nx.to_numpy_matrix(nxG)
    mask = np.zeros((N_max, 1))
    mask[:n] = 1
    tlpg, mapping = AM2tlp(AM, mask)
    return tlpg, mapping, mask

def mm2tlp(filepath, N_max):
    AM = np.zeros((N_max, N_max))
    mask = np.zeros((N_max, 1))
    mm_AM = csr_matrix.todense(mmread(filepath))
    n = len(mm_AM)
    assert n <= N_max

    AM[:n, :n] = mm_AM
    mask[:n] = 1
    tlpg, mapping = AM2tlp(AM, mask)
    return tlpg, mapping

def applyTlpLayoutAlgorithm(g, algo, propertyName, params_modifs={}):
    params = tlp.getDefaultPluginParameters(algo, g)
    for k, v in params_modifs.items():
        params[k] = v
    res = g.getLayoutProperty(propertyName)
    success, string = g.applyLayoutAlgorithm(algo, res, params)
    if(not success):
        return "fail layout"
    return res

def applyTlpAlgorithm(g, algo, propertyName, params_modifs={}):
    params = tlp.getDefaultPluginParameters(algo, g)
    for k, v in params_modifs.items():
        params[k] = v
    res = g.getDoubleProperty(propertyName)
    success = g.applyDoubleAlgorithm(algo, res, params)
    return res


def graph2predictData(g, graph_id, N_max, max_deg, model_inputs, features_names, topology_fn, swap=True, scalers=[]):
    AM = graph2AM(g, increased=N_max)
    DM = None
    perm = None
    real_n = g.numberOfNodes()
    N = max(N_max, real_n)
    input_dic = {}
    fake_inputs = True # no need for DM and sigma at prediction time
    if("DM" in model_inputs):
        if(fake_inputs):
            DM = np.zeros_like(AM)
        else:
            DM = AM2DM(AM)
        input_dic["DM"] = DM
    if("sigma" in model_inputs):
        if(fake_inputs):
            sigma = np.zeros((N, 1))
        else:
            if(DM is None):
                DM = AM2DM(AM)
            sigma =  getSigma(DM, real_n)
        input_dic["sigma"] =sigma

    #Time from now
    start = time.time()
    if("nodesMask" in model_inputs):
        mask = np.zeros((N,1))
        mask[:real_n] = 1
        input_dic["nodesMask"] = mask
    if("features" in model_inputs):
        F = graphFeatures(g, features_names, increased=N_max)
        if(scalers is not None and len(scalers) > 0):
            assert len(scalers) == F.shape[-1], f"Expected {F.shape[-1]} scalers but got {len(scalers)}"
            F = scaleFeatures(F, scalers)
        input_dic["features"] = F
    if("supports" in model_inputs):
        cur_AM = AM
        if(swap):
            cur_AM, _, perm = swapMatrix(AM, real_n=real_n, permutation=perm)
        input_dic["supports"] = topology_fn(cur_AM, max_deg)    
        # input_dic["supports"] = AM2customSupport(cur_AM)    
    end = time.time()
    elapsed = end - start
    elapsed_ms = elapsed * 1000
    
    #swap if necessary
    if(swap):
        swapperFn = {
            "DM":swapMatrix,
            "features":swapVector,
            "sigma":swapVector,
            "nodesMask":swapVector
        }
        for input_name in model_inputs:
            if(input_name != "supports"):
                swapped_input, _, perm = swapperFn[input_name](input_dic[input_name], real_n=real_n, permutation=perm)
                input_dic[input_name] = swapped_input

    #order inputs
    inputs = []
    for input_name in model_inputs:
        inputs.append(input_dic[input_name])
    return inputs, elapsed_ms