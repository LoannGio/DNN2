import numpy as np
import networkx as nx
import glob, os, random
import code.graph_preprocessing as preprocess
import code.spark_preprocessing as spreprocess

data_prefix = "./data/custom_graphs"

def dodecahedron():
    dodec = nx.dodecahedral_graph()
    N = dodec.number_of_nodes()
    tlpg, mapping, mask = preprocess.nx2tlp(dodec, N)
    return tlpg, "Dodecahedron"

def miserables_VH():
    miserables_g = spreprocess.load_graph(data_prefix+"/lesmiserables.gml")
    return miserables_g, "Miserables"

def grid(width=5, height=5):
    n = width*height
    grid_g = spreprocess.gen_one_graph("Grid", {"width":width, "height":height}, N_min=n, N_max=n)
    return grid_g, "Grid"

def complete(n=25):
    complete_g = spreprocess.gen_one_graph("Complete General Graph", {"nodes":n}, N_min=n, N_max=n)
    return complete_g, "Complete"

def hypercube(n_dim=3):
    hcube_g = nx.hypercube_graph(n_dim)
    N=hcube_g.number_of_nodes()
    tlpg, mapping, mask = preprocess.nx2tlp(hcube_g, N)
    return tlpg, "Hypercube"

def cycle(n=10):
    cycle_AM = np.zeros((n,n))
    for i in range(n-1):
        cycle_AM[i][i+1] = 1
    cycle_AM[0][n-1] = 1
    cycle_AM = np.logical_or(cycle_AM, cycle_AM.T)
    mask = np.ones((n, 1))
    tlpg, mapping = preprocess.AM2tlp(cycle_AM, mask)
    return tlpg, "Cycle"

def dwt_72():
    tlpg, mapping = preprocess.mm2tlp(data_prefix+"/dwt_72.mtx", 72)
    return tlpg, "dwt_72"

def can_96():
    tlpg, mapping = preprocess.mm2tlp(data_prefix+"/can_96.mtx", 96)
    return tlpg, "can_96"

def LFR_file(idx=None):
    p = "./data/LFR"
    f =""
    if(idx is None):
        files = glob.glob(os.path.join(p, "*"))
        random.shuffle(files)
        f = files[0]
        idx = os.path.basename(f).split(".")[0].split("_")[1]
    else:
        f = os.path.join(p, "graph_"+str(idx)+".tlpb.gz")
    tlpg = spreprocess.load_graph(f)
    return tlpg, "LFR_"+str(idx)

def ROME_file(path):
    g = spreprocess.load_graph(path)
    gname = os.path.basename(path).split(".")[0]
    return g, "ROME_"+gname

def star(n=10):
    AM = np.zeros((n,n))
    center_id = 0
    for i in range(n):
        if(i != center_id):
            AM[center_id][i] = 1
            AM[i][center_id] = 1
    tlpg, mapping = preprocess.AM2tlp(AM, np.ones((n,1)))
    return tlpg, "Star"