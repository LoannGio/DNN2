from tensorflow.keras import backend as K
import tensorflow as tf

dtype = "float32"
epsilon = 1e-16


def stress(y_pred, DM, real_nodes_mask, wij_alpha=2):
    # y_pred: model predictions, ie. graph layout
    pred_pairwise_vectors = K.expand_dims(y_pred, 1) - K.expand_dims(y_pred, 2)
    pred_pairwise_squared_dist = tf.cast(K.sum(K.square(pred_pairwise_vectors), axis=-1), dtype)

    pred_distanceMatrix = K.cast(tf.math.sqrt(pred_pairwise_squared_dist+1e-8), dtype)
    GT_distanceMatrix = K.cast(tf.identity(DM), dtype)
    real_nodes_mask_vector = K.cast(tf.identity(real_nodes_mask), dtype)
    real_nodes_mask_matrix = real_nodes_mask_vector * tf.transpose(real_nodes_mask_vector, perm=(0, 2, 1))  # vector mask to matrix

    GT_distanceMatrix = GT_distanceMatrix*real_nodes_mask_matrix  # set to 0 all distances with fictive nodes
    pred_distanceMatrix = pred_distanceMatrix*real_nodes_mask_matrix  # set to 0 all distances with fictive nodes

    real_n_nodes = K.sum(K.cast(real_nodes_mask_vector, dtype), axis=(1, 2))  # number of real nodes, used for normalization

    ############# Stress computation #############
    sub = GT_distanceMatrix - pred_distanceMatrix
    squared = K.square(sub)

    # using pow with negative power and 0.. causes infinity: need a matrix without zeros to compute power: replace by 1.
    GT_distanceMatrix_no_zeros = tf.where(GT_distanceMatrix < epsilon, tf.ones_like(GT_distanceMatrix), GT_distanceMatrix)

    W = tf.where(GT_distanceMatrix < epsilon, tf.zeros_like(GT_distanceMatrix), GT_distanceMatrix_no_zeros**-wij_alpha)

    stress_val = W*squared

    ############# Normalization #############
    normalized_stress = K.cast(K.sum(stress_val, axis=(1, 2)), dtype) / K.cast(K.square(real_n_nodes), dtype)
    return normalized_stress


def tsnet(y_pred, DM, real_nodes_mask, sigmas, n_nodes, l_kl=None, l_c=None, l_r=None, r_eps=0.05, epsilon=1e-16, batch_size=8):
    #from https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.13187?casa_token=WmGma2M1UcYAAAAA:xOzGocWGf0_N_xQa4V1AG6KCYSR5walQQfphyTEpWPh-AuKPSGaHlYK3sYIsNGcviUJm9efOC0sP9h9GLg
    #git https://github.com/HanKruiger/tsNET
    def p_ij_conditional_var(X, sigma, N, matrix_mask):
        sqdistance = X**2
        esqdistance = tf.math.exp(-sqdistance / tf.reshape((2 * (sigma**2)), (batch_size, N, 1)))
        esqdistance = K.cast(esqdistance, dtype)* matrix_mask
        esqdistance_zd = tf.linalg.set_diag(esqdistance, tf.zeros((batch_size,N)))# or: np.zeros((esqdistance.shape[?]))
        row_sum = tf.reshape(K.sum(esqdistance_zd, axis=2), (batch_size,N, 1))
        zeros = tf.zeros(row_sum.shape)
        res = tf.math.divide_no_nan(esqdistance_zd,row_sum) 
        return res
    
    def p_ij_sym_var(p_ij_conditional, real_n):
        num = (p_ij_conditional + tf.transpose(p_ij_conditional, perm=(0,2,1)))
        denom = (2 * real_n)
        res = num / denom
        return  res

    def sqeuclidean_var(X, matrix_mask):
        pred_pairwise_diff = K.expand_dims(X, 1) - K.expand_dims(X, 2)
        pred_pairwise_squared_dist = tf.cast(K.sum(K.square(pred_pairwise_diff), axis=-1), dtype)
        pred_pairwise_squared_dist = pred_pairwise_squared_dist * matrix_mask
        return pred_pairwise_squared_dist

    def q_ij_student_t_var(Y, N, matrix_mask):
        sqdistance = sqeuclidean_var(Y, matrix_mask)
        one_over = tf.linalg.set_diag(tf.math.divide_no_nan(1.0 , (sqdistance + 1.0)), tf.zeros((batch_size,N)))
        one_over = K.cast(one_over, dtype) * K.cast(matrix_mask, dtype)
        #denom = tf.reshape(K.sum(one_over, axis=(1,2)), (batch_size, 1,1))
        denom = tf.reshape(K.sum(one_over, axis=(1,2)), (batch_size, 1,1))
        res = one_over / denom
        return res

    def euclidean_var(X, matrix_mask):
        res = K.maximum(sqeuclidean_var(X, matrix_mask), epsilon) ** 0.5
        res = res * matrix_mask
        return res

    X_DM = K.cast(tf.identity(DM), dtype)
    local_real_nodes_mask = K.cast(tf.identity(real_nodes_mask), dtype)
    matrix_real_nodes_mask = local_real_nodes_mask * tf.transpose(local_real_nodes_mask, perm=(0,2,1))
    X_DM = X_DM*matrix_real_nodes_mask 
    Y_pos = y_pred*local_real_nodes_mask 

    N = n_nodes
    real_n = tf.reshape(K.cast(tf.round(K.sum(local_real_nodes_mask,axis=(1,2))), dtype), (batch_size, 1, 1))

    sigma =K.cast(tf.identity(sigmas), "float32") 
    p_ij_conditional = p_ij_conditional_var(X_DM, sigma, N, matrix_real_nodes_mask)
    p_ij = p_ij_sym_var(p_ij_conditional, real_n)
    q_ij = q_ij_student_t_var(Y_pos, N, matrix_real_nodes_mask)
    p_ij_safe = K.maximum(p_ij, epsilon)
    q_ij_safe = K.maximum(q_ij, epsilon)

    # Kullback-Leibler term
    kl = K.sum(p_ij * K.log(p_ij_safe / q_ij_safe), axis=2)
    # Compression term
    compression = tf.reshape(K.cast((1 / (2 * real_n)), dtype), (batch_size, 1)) * K.cast(K.sum(Y_pos**2, axis=2), dtype)
    # Repulsion term       
    repulsion = tf.reshape(K.cast(-(1 / (2 * real_n**2)), dtype), (batch_size,1)) * K.sum(tf.linalg.set_diag(K.log(euclidean_var(Y_pos,matrix_real_nodes_mask)+r_eps), tf.zeros((batch_size,N)))*matrix_real_nodes_mask, axis=2)
    
    cost = kl * l_kl + compression * l_c + repulsion * l_r 

    return K.sum(cost, axis=-1)