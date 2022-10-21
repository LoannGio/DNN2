from tensorflow.keras.layers import Input, BatchNormalization, Add, Dense, Activation
from tensorflow.keras import Model
from .graph_layers import GraphConvolution

def conv_block_residual(input, support, filters, supports, inputXmask, conv_shortcut=False, activation="relu"):
    features = input
    reg=None
    for i in range(len(filters)):
        filter_i = filters[i]
        x = GraphConvolution(units=filter_i, support=support, activation=activation, kernel_regularizer=reg)([features, supports])
        x = BatchNormalization()(x)
        features = x
    
    shortcut = None
    if(conv_shortcut):
        shortcut = GraphConvolution(units=filters[-1], support=support, activation=activation, kernel_regularizer=reg)([input, supports])
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    x = x * inputXmask
    return x


def build_model(Nmax, initial_features_vector_size, support_size, loss_is_tsnet=False):
    if(Nmax == -1):
        Nmax = None
    inputDM = Input(shape=(Nmax, Nmax), name="input_DM")  # required for loss computation, training stage only
    inputS = Input(shape=(support_size, Nmax, Nmax), name="input_supports") # encodes graph topology, Spectrum or Spatia
    inputX = Input(shape=(Nmax, initial_features_vector_size), name="input_features") # initial graph signal, aka nodes Features
    inputXmask = Input(shape=(Nmax,1), name="input_nodesMask") # mask of real/fictive nodes after padding to Nmax
    inputs = [inputDM, inputXmask, inputX, inputS]
    if(loss_is_tsnet):
        inputSigma = Input(shape=(Nmax,1), name="input_sigma") #required for loss computation, training stage only
        inputs.append(inputSigma)

    last_support = min(2, support_size)
    x = inputX * inputXmask
    
    #Features extraction
    x = conv_block_residual(x, last_support, [16, 16, 32], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [16, 16, 32], inputS, inputXmask)
    stack1 = conv_block_residual(x, support_size, [16, 16, 32], inputS, inputXmask)

    x = conv_block_residual(stack1, last_support, [32, 32, 64], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)
    stack2 = conv_block_residual(x, support_size, [32, 32, 64], inputS, inputXmask)

    x = conv_block_residual(stack2, last_support, [64, 64, 128], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    x = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)
    stack3 = conv_block_residual(x, support_size, [64, 64, 128], inputS, inputXmask)

    x = conv_block_residual(stack3, last_support, [128, 128, 128], inputS, inputXmask, conv_shortcut=True)
    x = conv_block_residual(x, last_support, [128, 128, 128], inputS, inputXmask)
    stack4 = conv_block_residual(x, last_support, [128, 128, 128], inputS, inputXmask)

    #Regression
    x = stack4
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(2, name="output")(x)
    
    return Model(inputs, x)

if __name__ == "__main__":
    #Example building model
    Nmax = 128
    initial_features_vector_size = 2 #e.g., [id, random] for each node
    support_size = 4 # K, the order of Chebyshev Filter
    model = build_model(Nmax, initial_features_vector_size, support_size, loss_is_tsnet=False)
    model.summary()
