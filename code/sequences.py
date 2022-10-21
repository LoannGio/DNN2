import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler
import code.spark_preprocessing as spreprocess
import random

class DNN2Sequence(Sequence):
    def __init__(self, batch_size, features_names, n_features, max_deg, model_inputs, topology_fn, swap=True, files=None, N_max=-1, scalers=None, doSeed=False, normalizeFeatures=True, num_worker_thread="*"):
        self.batch_size = batch_size
        self.data = []
        self.features_names = features_names
        self.files = files
        self.N_max = N_max
        self.max_deg = max_deg
        self.model_inputs = model_inputs
        self.swap = swap
        self.doSeed = doSeed
        self.n_features = n_features
        self.topology_fn = topology_fn
        
        sc = self.spark_load_data(num_worker_thread) # loads graphs data into 'self.data'
        spreprocess.stop_spark(sc)
       
        self.scalers = None
        if("features" in model_inputs and normalizeFeatures == True):
            self.scalers = self.normalize_features(scalers)

    def normalize_features(self, scalers=None):
        n_features = self.n_features
        F = np.empty((len(self.data), self.N_max, n_features))
        for i in range(len(self.data)):
            F[i] = self.data[i][0][0][self.model_inputs.index("features")]

        fit = False
        if(scalers is None):
            scalers = [MinMaxScaler(feature_range=(0,1)) for i in range(n_features)]
            fit = True

        for f in range(n_features):
            fs = F[:,:,f].astype(float)
            fs = fs.flatten()
            fs = fs.reshape((*fs.shape, 1))            
            if(fit):
                fs = scalers[f].fit_transform(fs)
            else:
                fs = scalers[f].transform(fs)
            F[:,:,f] = fs.reshape(F[:,:,f].shape)
        for i in range(len(self.data)):
           self.data[i][0][0][self.model_inputs.index("features")][:,:] = F[i]
        return scalers

    def spark_load_data(self, num_worker_thread="*"):
        sc, graphs_rdd = spreprocess.load_many_graphs(self.files, self.batch_size, num_worker_thread)    
        self.data = spreprocess.rdd2trainingData(graphs_rdd, self.doSeed, self.N_max, self.max_deg, self.model_inputs, self.features_names, self.topology_fn, self.swap)
        return sc

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = [[] for i in range(len(self.model_inputs))]
        labels = []
        for b, d_realN in enumerate(batch):
            d = d_realN[0]
            labels.append(d[1])
            model_inputs = d[0]
            for i,model_input in enumerate(model_inputs):
                inputs[i].append(model_input)
        for i in range(len(inputs)):
            inputs[i] = np.array(inputs[i])

        return inputs, labels

    def get_random_samples(self, n):
        inputs = [[] for i in range(len(self.model_inputs))]
        labels = []
        real_Ns = []
        for i in range(n):
            d_realN = self.data[random.randint(0, len(self.data)-1)]
            d = d_realN[0]
            real_Ns.append(d_realN[1])
            labels.append(d[1])
            model_inputs = d[0]
            for i,model_input in enumerate(model_inputs):
                inputs[i].append(model_input)
        for i in range(len(inputs)):
            inputs[i] = np.array(inputs[i])
        return (inputs, labels), real_Ns

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
