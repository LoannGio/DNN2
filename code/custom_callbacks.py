import tensorflow as tf
import os
import shutil
import random
import numpy as np
import warnings
from matplotlib import pyplot as plt
import networkx as nx
from .metrics import get_nodes_diameter
import code.SOTA_layouts as SOTA


class VisualCallback(tf.keras.callbacks.Callback):
    # At the end of each epoch, draws "n_tests" graphs from "val_sequence" with "drawer" algorithm and current model ; save the result in "saveDir"
    def __init__(self,val_sequence, drawer_instance=SOTA.S_GD2(), saveDir="./callback_tests", n_tests=5):
        super(VisualCallback, self).__init__()
        self.n_tests = n_tests
        self.saveDir = saveDir
        self.sequence = val_sequence
        self.N_max = val_sequence.N_max
        self.model_inputs = val_sequence.model_inputs
        self.GTdrawer_instance = drawer_instance
        if(os.path.exists(self.saveDir)):
            shutil.rmtree(self.saveDir)
        os.makedirs(self.saveDir)


    def on_epoch_end(self, epoch, logs={}):
        warnings.filterwarnings("ignore")
        (random_samples, real_nodes) = self.sequence.get_random_samples(self.sequence.batch_size)
        random_samples_features = random_samples[0] #[1] is label
        preds = self.model.predict(random_samples_features, batch_size=self.sequence.batch_size)
        saveDir = self.saveDir

        figW=10
        figH=figW
        plt.rcParams["figure.figsize"] = (figW, figH)
        done = []
        for i in range(self.n_tests):
            n = random.randint(0, self.sequence.batch_size-1)
            while(n in done):
                n = random.randint(0, self.sequence.batch_size-1)
            done.append(n)
            fig, axes = plt.subplots(nrows=2, ncols=2)            

            DM = random_samples_features[self.model_inputs.index("DM")][n]
            real_n = real_nodes[n]
            N = max(self.N_max, real_n)
            R = fig.get_dpi()*figW
            nodeSize = int(max(1, get_nodes_diameter(R, real_n)))
            if("nodesMask" in self.model_inputs):
                nodes_mask = random_samples_features[self.model_inputs.index("nodesMask")][n]
            else:
                nodes_mask = np.ones((N, 1))

            axes[0,0].matshow(DM == 1)
            axes[0,0].set_title("Adjacency Matrix")

            xs = np.zeros((N))
            ys = np.zeros((N))
            for i in range(N):
                if(nodes_mask[i] == 1):
                    xs[i] = preds[n][i][0]
                    ys[i] = preds[n][i][1]

            G = nx.Graph()
            pos = {}
            for i in range(N):
                if(nodes_mask[i] == 1):
                    G.add_node(i)
                    pos[i] = (xs[i], ys[i])

            for i in range(N):
                for j in range(N):
                    if(DM[i][j] == 1):
                        G.add_edge(i, j)
                        G.add_edge(j, i)
            edge_alpha=0.1
            nx.drawing.nx_pylab.draw_networkx_nodes(G, pos, node_size=nodeSize, ax=axes[1,1], node_color="r")
            nx.drawing.nx_pylab.draw_networkx_edges(G, pos,alpha=edge_alpha, ax=axes[1,1])
            axes[1,1].set_title("predicted positions + original edges")
            axes[1,1].axis("off")
            
            posGT = self.GTdrawer_instance.layout(DM, nodes_mask)
            nx.draw(G, posGT, node_size=nodeSize, ax=axes[0,1], node_color="r", edge_color=(0,0,0,edge_alpha))
            axes[0,1].set_title("original graph with " + str(self.GTdrawer_instance.name))

            plt.savefig(saveDir+"/epoch_"+str(epoch+1)+"_res_"+str(n)+".pdf")
            plt.close()
