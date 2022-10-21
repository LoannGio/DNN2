import numpy as np
import networkx as nx
import sklearn.manifold as skmanifold
import s_gd2
from tulip import tlp
import time
import json
import os
import joblib
import code.models as models
import code.graph_preprocessing as preprocess

def upSamplePos(pos2d, mask):
    N_max = mask.shape[0]
    upsampled_pos = np.zeros((N_max, 2))
    real_i = 0
    for i in range(N_max):
        if(mask[i] == 1):
            upsampled_pos[i] = pos2d[real_i]
            real_i +=1
    return upsampled_pos

def layoutIsNotConstant(pos, mask):
    N = pos.shape[0]
    first = None
    for i in range(N):
        if(mask[i] == 1):
            if(first is None):
                first = pos[i]
            else:
                if(first[0] != pos[i][0] or first[1] != pos[i][1]):
                    return True
    return False

class ILayout:
    def __init__(self):
        super().__init__()
        self.execTimes = []

    def get_start(self):
        start_timestamp = time.time()
        return start_timestamp
    
    def get_elapsed(self, start):
        end_timestamp = time.time()
        elapsed = end_timestamp - start
        ms_elapsed = elapsed * 1000
        return ms_elapsed

    def layout(self, DM, mask):
        raise NotImplementedError("This interface does not implement this method")

    def predict(self, sequence):
        preds = np.empty((len(sequence), sequence.N_max, 2))
        for i in range(len(sequence)):
            item = sequence[i]
            c_DM = item[0][sequence.model_inputs.index("DM")].squeeze(axis=0)
            c_mask = item[0][sequence.model_inputs.index("nodesMask")].squeeze(axis=0)
            preds[i] = self.layout(c_DM, c_mask)
        return preds     

class KamadaKawai(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "KamadaKawai"

    #see https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html
    def layout(self, DM, mask):
        N_max = DM.shape[0]
        G = preprocess.AM2nx(DM)
        startTime=self.get_start()
        lay = nx.drawing.layout.kamada_kawai_layout(G)
        self.execTimes.append(self.get_elapsed(startTime))
        pos2d = preprocess.nxPos2array(lay, N_max)
        return pos2d

class KamadaKawai_OGDF(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "Tree Radial"
    
    #see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    def layout(self, DM, mask):
        N_max = DM.shape[0]
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))

        propName = "viewLayout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "Random layout", propName)
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "Kamada Kawai (OGDF)", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class SMACOF(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "SMACOF"

    #see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html
    def layout(self, DM, mask):
        DM = preprocess.shrinkDM(DM, mask)
        startTime=self.get_start()
        (pos2d, stress) = skmanifold.smacof(DM, n_components=2)
        self.execTimes.append(self.get_elapsed(startTime))
        pos2d = upSamplePos(pos2d, mask)
        return pos2d
        
class S_GD2(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "S_GD2"

    # see https://github.com/jxz12/s_gd2
    def layout(self, DM, mask):
        DM = preprocess.shrinkDM(DM, mask)  
        I,J = preprocess.DM2edgesIdx(DM)
        I = I.astype("int32")
        J = J.astype("int32")
        startTime=self.get_start()
        pos2d = s_gd2.layout(I, J)
        self.execTimes.append(self.get_elapsed(startTime))
        pos2d = upSamplePos(pos2d, mask)
        return pos2d

class LinLog(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "LinLog"

    #see https://tulip.labri.fr/Documentation/current/tulip-python/html/tulippluginsdocumentation.html#linlog
    def layout(self, DM, mask):
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        N_max = DM.shape[0]
        propName = "linlog_layout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "LinLog", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        pos2d = np.zeros((N_max, 2))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class GEM(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "GEM"

    def layout(self, DM, mask):
        N_max = DM.shape[0]
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))

        propName = "gem_layout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "GEM (Frick)", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class GEM_OGDF(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "GEM_OGDF"

    def layout(self, DM, mask):
        N_max = DM.shape[0]
        real_n = int(np.sum(mask))
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))
        propName = "viewLayout"
        const = 30000
        params_modifs = {
            "Attraction formula":"GEM", 
            "number of rounds":max(4*(real_n**2), const)
        }

        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "Random layout", propName)
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, 'GEM Frick (OGDF)', propName, params_modifs=params_modifs)
        self.execTimes.append(self.get_elapsed(startTime))

        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class TSNE(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "TSNE"
    
    #see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    def layout(self, DM, mask):
        DM = preprocess.shrinkDM(DM, mask)
        startTime=self.get_start()
        pos2d = skmanifold.TSNE(n_components=2).fit_transform(DM)
        self.execTimes.append(self.get_elapsed(startTime))
        pos2d = upSamplePos(pos2d, mask)
        return pos2d

class PivotMDS(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "PivotMDS"
    
    def layout(self, DM, mask):
        N_max = DM.shape[0]
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))

        propName = "pivotmds_layout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "Pivot MDS (OGDF)", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class FM3(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "FM^3"
    
    def layout(self, DM, mask):
        N_max = DM.shape[0]
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))

        propName = "fm3_layout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "FM^3 (OGDF)", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class TreeRadial(ILayout):
    def __init__(self):
        super().__init__()
        self.name = "Tree Radial"
    
    def layout(self, DM, mask):
        N_max = DM.shape[0]
        tlpg, mapping = preprocess.AM2tlp(DM, mask)
        pos2d = np.zeros((N_max,2))

        propName = "treeRadial_layout"
        startTime=self.get_start()
        layout_prop=preprocess.applyTlpLayoutAlgorithm(tlpg, "Tree Radial", propName)
        self.execTimes.append(self.get_elapsed(startTime))
        for n in tlpg.getNodes():
            pos3d = layout_prop.getNodeValue(n)
            pos2d[mapping[n.id]] = [pos3d[0], pos3d[1]]
        return pos2d

class DNN2(ILayout):
    def __init__(self, h5path, jsonpath, N_max=-1):
        super().__init__()
        with open(jsonpath, "r") as f:
            self.modelinfos =  json.load(f)
        
        self.name = os.path.basename(h5path).split(".")[0]
        self.N_max = N_max
        self.modelinfos["model"] = models.build_model(self.N_max, self.modelinfos["initial_features_vector_size"], self.modelinfos["max_deg"]+1, self.modelinfos["tsnet_loss"])
        self.modelinfos["model"].load_weights(h5path)
        self.modelinfos["inputs"] = [i.name.split(":")[0].split("_")[1] for i in self.modelinfos["model"].inputs]
        self.modelinfos["topology_fn"] = getattr(preprocess, self.modelinfos["topology"])
        if("scalersPath" in self.modelinfos.keys()):
            self.modelinfos["scalers"] = joblib.load(self.modelinfos["scalersPath"])
        else:
            scalersPath = os.path.join(h5path.replace(os.path.basename(h5path), ""), "scalers.pkl")
            self.modelinfos["scalers"] = joblib.load(scalersPath)

    #see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html
    def layout(self, g):
        i=0
        max_deg = self.modelinfos["max_deg"]
        model_inputs = self.modelinfos["inputs"]
        nodes_features = self.modelinfos["features"]
        scalers = self.modelinfos["scalers"]
        
        startTime=self.get_start()
        inputs, time_preprocess = preprocess.graph2predictData(g, i, self.N_max, max_deg, model_inputs, nodes_features, self.modelinfos["topology_fn"], swap=False,scalers=scalers)

        batched_input = [[] for _ in range(len(model_inputs))]
        for i in range(len(inputs)):
            batched_input[i].append(inputs[i])
        for i in range(len(batched_input)):
            batched_input[i] = np.array(batched_input[i])
        
        pred = self.modelinfos["model"](batched_input, training=False) 
        exec_time = self.get_elapsed(startTime)
        self.execTimes.append(exec_time + time_preprocess)
        if(type(pred) == np.ndarray):
            pred = pred.squeeze()
        else:
            pred = pred.numpy().squeeze()
        return pred



methods = [
    "S_GD2",
    "KamadaKawai",
    "KamadaKawai_OGDF",
    "SMACOF",
    "LinLog",
    "GEM",
    "GEM_OGDF",
    "TSNE",
    "PivotMDS",
    "FM3",
    "TreeRadial",
]