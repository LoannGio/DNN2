from code.DNN2_Wrapper import DNN2_Wrapper
import code.graph_preprocessing as preprocess
import code.models as models
import glob
import os
import random
import tensorflow as tf

def load_dir(path, extension=".tlpb.gz", shuffle=True):
    files = glob.glob(os.path.join(path, "*"+extension))
    if(shuffle):
        random.shuffle(files)
    return files

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" # remove INFO logs 
os.environ["OMP_NUM_THREADS"] = "1" # prevents tulip from multi threading, which would create conflicts with spark and slow data preprocessing

#GPU SETUP (in case there are several GPUs)
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_id], True) # in case you don't want to reserv ALL the GPU memory

execName = "DNN2_implementation_example"
Nmax = 128 #Every graph will be padded with fictive nodes up to Nmax nodes
batch_size = 32

#Nodes features "names", will be generated later
features = [
    "id",
    "random:0",
    #"random:1" # <--- you can have several random that way
    "LAYOUT:Pivot MDS (OGDF)" # to use a layout as input feature, give : "LAYOUT:XXXXXX" where XXXXXX is a layout algorithm of tulip-python (https://tulip.labri.fr/Documentation/current/tulip-python/html/tulippluginsdocumentation.html#layout)
]

K = 4
topology_data_structure_fn = preprocess.AM2Chebyshev

# K = 0 #GCN filters have no depth
# topology_data_structure_fn = preprocess.AM2GCNfilter

model_fn = models.build_model # wrapper will assume model_fn signature
loss_name = "tsnet" # "tsnet" or "stress"

#Setup model and data generation parameters
model_weights_path = "" # used for transfer learning (ie. Finetune instances) and/or for tsNET training (which is done in 2 stages, 1 training for each stage -> stage 2 weights are set to that of the best stage 1 epoch)
wrapper = DNN2_Wrapper(execName, Nmax, features, batch_size, topology_data_structure_fn, K, model_fn, loss_name, model_weights_path=model_weights_path)

#Generate/load graphs into structures that can be fed to the model
train_files = load_dir("./data/rome_graphs/train", extension=".tlpb.gz")
val_files = load_dir("./data/rome_graphs/val", extension=".tlpb.gz")
wrapper.prepare_data(train_files, val_files, num_worker_thread="*") # num_worker_thread = number of worker threads for parallel execution. * means all (ie. number of logical cores)

#Train the model
saveTrainingPrefix = "."
wrapper.train(saveTrainingPrefix)

#Save training infos for traceability
wrapper.save_trainInfos()

print("DONE")