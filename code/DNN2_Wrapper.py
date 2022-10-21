from tensorflow.keras.optimizers import Adam
import code.losses as losses
import code.sequences as sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from code.custom_callbacks import VisualCallback

import joblib
import json
import os
import glob
import shutil

class DNN2_Wrapper:
    def __init__(self, exec_name, Nmax, graph_features, batch_size, topology_fn, K, model_fn, loss_name="stress", opt=Adam(learning_rate=0.001), model_weights_path=""):
        self.exec_name = exec_name

        #Graph
        self.Nmax = Nmax
        self.graph_features = graph_features
        self.initial_features_size = len(graph_features)
        for feature in graph_features:
            if(feature.startswith("LAYOUT:")):
                self.initial_features_size += 1 # because a layout gives 2 features per node -> X and Y coordinate
        self.topology_fn = topology_fn
        self.K = K
               
        #DNN
        self.batch_size = batch_size
        self.loss_is_tsnet = loss_name == "tsnet"
        loss_fn = None
        self.support_size = K+1
        self.model = model_fn(Nmax, self.initial_features_size, self.support_size, loss_is_tsnet=self.loss_is_tsnet)
        self.model.summary()
        if(os.path.exists(model_weights_path)):
            self.model.load_weights(model_weights_path)
            print("WEIGHTS LOADED FROM ", model_weights_path, "\n")
        self.model_inputs = [i.name.split(":")[0].split("_")[1] for i in self.model.inputs]
        if(loss_name == "tsnet"):
            l_kl, l_c, l_r = 1., 1.2, 0.      # lambda for tsnet stage 1
            #l_kl, l_c, l_r = 1., 0.01, 0.6   # lambda for tsnet stage 2
            #l_kl, l_c, l_r = 1., 0.1, 0.1   # lambda for tsnet* stage 1
            #l_kl, l_c, l_r = 0.4, 0.01, 1.1  # lambda for tsnet* stage 2
            r_eps=0.05
            l_sum = l_kl + l_c + l_r
            l_kl /= l_sum
            l_c /= l_sum
            l_r /= l_sum
            loss_fn = losses.tsnet(
                self.model.output, 
                self.model.input[self.model_inputs.index("DM")], 
                self.model.input[self.model_inputs.index("nodesMask")], 
                self.model.input[self.model_inputs.index("sigma")], 
                Nmax, 
                batch_size=batch_size, 
                l_kl=l_kl, l_c=l_c, l_r=l_r, r_eps=r_eps
            )
        elif(loss_name == "stress"):
            loss_fn = losses.stress(
                self.model.output, 
                self.model.input[self.model_inputs.index("DM")],
                self.model.input[self.model_inputs.index("nodesMask")]
            )

        self.model.add_loss(loss_fn)
        self.model.compile(optimizer=opt)

    def prepare_data(self, train_files, val_files, swap_fictive_nodes=True, normalizeFeatures=True, num_worker_thread="*"):
        scalers = None
        
        print("============ Build Train sequence ============")
        self.train_sequence = sequences.DNN2Sequence(
            self.batch_size, 
            self.graph_features, 
            self.initial_features_size, 
            self.K, 
            self.model_inputs, 
            self.topology_fn, 
            swap=swap_fictive_nodes, 
            files=train_files, 
            N_max=self.Nmax, 
            normalizeFeatures=normalizeFeatures,
            num_worker_thread=num_worker_thread
        )
        if(normalizeFeatures):
            #validation data should be scaled based on train scalers (same for test data)
            scalers = self.train_sequence.scalers 

        print("============ Build Validation sequence ============")
        self.val_sequence = sequences.DNN2Sequence(
            self.batch_size, 
            self.graph_features, 
            self.initial_features_size, 
            self.K, 
            self.model_inputs, 
            self.topology_fn, 
            swap=swap_fictive_nodes, 
            files=val_files, 
            N_max=self.Nmax,
            normalizeFeatures=normalizeFeatures, 
            num_worker_thread=num_worker_thread,
            scalers=scalers
        )
        self.scalers = scalers

    def train(self, dirs_prefix, patience=20, max_epochs=200, monitoring="val_loss", visualCallback=True):
        self.execDir = os.path.join(dirs_prefix, self.exec_name)
        while(os.path.exists(self.execDir)): # in case we forgot to change execName, don't want to accidently erase previous data
            self.execDir+="_new"
        self.patience = patience

        modelsDir = os.path.join(self.execDir, "models_h5")
        os.makedirs(modelsDir)  
        fname = modelsDir + "/"+self.exec_name+"_{epoch:02d}.h5"

        callbacks = []
        if(visualCallback):
            visualCallbackDir = os.path.join(self.execDir, "visualCallback")
            callbacks.append(VisualCallback(self.val_sequence, saveDir=visualCallbackDir, n_tests=3))
        callbacks.append(ModelCheckpoint(fname, monitor=monitoring, verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1))
        callbacks.append(EarlyStopping(monitor=monitoring, min_delta=0, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=True))

        self.history = self.model.fit(
            x=self.train_sequence,
            validation_data=self.val_sequence,
            epochs=max_epochs,
            callbacks=callbacks
        )

    def save_trainInfos(self):
        # could simply dump the instance of DNN2_Wrapper
        js = { 
            "execName": self.exec_name,
            "batch_size": self.batch_size,
            "features": self.graph_features,
            "initial_features_vector_size": self.initial_features_size,
            "topology" : self.topology_fn.__name__,
            "Nmax": self.Nmax,
            "max_deg":self.K,
            "patience": self.patience,
            "tsnet_loss":self.loss_is_tsnet,
            "history": {},
        }
        normalizedFeatures = False
        if(self.scalers is not None):
            normalizedFeatures = True
            scalersPath = os.path.join(self.execDir, "scalers.pkl")
            joblib.dump(self.scalers, scalersPath)
            js["scalersPath"] = os.path.abspath(scalersPath)
        js["normalizedFeatures"] = normalizedFeatures
        
        for (k,v) in self.history.history.items():
            js["history"][k] = v

        with open(os.path.join(self.execDir, "trainInfos.json"), "w") as f:
            json.dump(js, f)
            f.close()