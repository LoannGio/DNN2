from pyspark import SparkContext, SparkConf
from tulip import tlp
import numpy as np

import code.graph_preprocessing as preprocess

import os
import random

def init_spark(num_worker_thread):
    conf = SparkConf().setMaster(f"local[{num_worker_thread}]").set("spark.executor.memory", "500m").set("spark.driver.memory", "25g").set("spark.driver.maxResultSize", "25g")
    sc = SparkContext(appName="TLPGraphsGen", conf=conf)
    return sc

def stop_spark(sc):
    for (id, rdd) in sc._jsc.getPersistentRDDs().items():
        rdd.unpersist()
    sc.stop()

def load_graph(filepath):
    g = tlp.loadGraph(filepath)
    if(not tlp.ConnectedTest.isConnected(g)):
        params = tlp.getDefaultPlpreprocessinParameters('Make Connected', g)
        success = g.applyAlgorithm('Make Connected', params)
    return g

def load_many_graphs(files, batch_size, num_worker_thread, n_graphs=-1):
    sc = init_spark(num_worker_thread)
    while(not ((len(files) % batch_size) == 0)):
        files.append(files[random.randint(0, len(files)-1)])

    files_tuples = []
    for i, f in enumerate(files):
        files_tuples.append((i, f))

    filenames_rdd = sc.parallelize(files_tuples).repartition(200)
    graphs_rdd = filenames_rdd.map(lambda i_f: (i_f[0],load_graph(i_f[1])))
    return sc, graphs_rdd

def save_graph(g, id, save_dir, extension):
    filename = "graph_"+str(id)
    full_path = os.path.join(save_dir, filename)+extension
    success = tlp.saveGraph(g, full_path)
    return success

def gen_one_graph(graphGen, params, N_min, N_max):
    g = tlp.importGraph(graphGen, params)
    while(not graphIsValid(g, N_min, N_max)):
        g = tlp.importGraph(graphGen, params)  
    return g

def graphIsValid(g, N_min, N_max):
    if(N_min is not None and g.numberOfNodes() < N_min):
        return False
    if(N_max is not None and g.numberOfNodes() > N_max):
        return False
    if(not tlp.ConnectedTest.isConnected(g)):
        return False
    return True

def graph2trainingData(g, graph_id, doSeed, N_max, K, model_inputs, features_names, topology_fn, swap=True):
    real_n_nodes = g.numberOfNodes()
    AM = preprocess.graph2AM(g, increased=N_max)
    label = np.random.rand(AM.shape[0], 2) # never used, just has to fit output shape, technical need to be there
    inputs=  []
    DM = None
    perm = None
    nodes_mask =None
    if(doSeed):
        np.random.seed(graph_id)

    for i in model_inputs:
        if(i == "DM"):
            DM = preprocess.AM2DM(AM, fill_value=-1)
            if(swap):
                swapped_DM, _, perm = preprocess.swapMatrix(DM, real_n=real_n_nodes, permutation=perm)                
                inputs.append(swapped_DM)
            else:
                inputs.append(DM)
        elif(i == "features"):
            features = preprocess.graphFeatures(g, features_names, increased=AM.shape[0])
            if(swap):
                swapped_features, nodes_mask, perm = preprocess.swapVector(features, real_n=real_n_nodes, permutation=perm)
                inputs.append(swapped_features)
            else:
                inputs.append(features)
        elif(i == "sigma"):
            if(DM is None):
                DM = preprocess.AM2DM(AM, fill_value=-1)
            sigma = preprocess.getSigma(DM, real_n_nodes)
            if(swap):
                sigma, nodes_mask, perm = preprocess.swapVector(sigma, real_n=real_n_nodes, permutation=perm)
            inputs.append(sigma)
        elif(i == "supports"):
            cur_AM = AM
            if(swap):
                cur_AM, _, perm = preprocess.swapMatrix(AM, real_n=real_n_nodes, permutation=perm)
            supports = topology_fn(cur_AM, K)
            inputs.append(supports)
        elif(i == "nodesMask"):
            cur_nodes_mask = None
            if(swap):
                if(nodes_mask is not None):
                    cur_nodes_mask = nodes_mask
                else:
                    cur_nodes_mask = np.zeros((AM.shape[0], 1))
                    cur_nodes_mask[:real_n_nodes] = 1
                    cur_nodes_mask, _, perm = preprocess.swapVector(cur_nodes_mask, real_n=real_n_nodes, permutation=perm)
            else:
                cur_nodes_mask = np.zeros((AM.shape[0], 1))
                cur_nodes_mask[:real_n_nodes] = 1
            inputs.append(cur_nodes_mask)
    g = None
    inputs = tuple(inputs)
    return (inputs, label), real_n_nodes


def rdd2trainingData(graphs_rdd, doSeed, N_max, max_deg, model_inputs, features_names, topology_fn, swap=True):
    trainingData_rdd = graphs_rdd.map(lambda i_g: graph2trainingData(i_g[1], i_g[0], doSeed, N_max, max_deg, model_inputs, features_names,topology_fn, swap=swap))
    trainingData = rdd2list(trainingData_rdd)
    return trainingData

def rdd2list(rdd):
    rdd.cache().count()
    l = list(rdd.toLocalIterator())
    rdd.unpersist()
    return l
