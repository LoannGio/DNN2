import numpy as np
import math
from tulip import tlp
from sklearn.preprocessing import MinMaxScaler
import warnings
from .graph_preprocessing import AM2tlp,pos2DM, shrinkDM, applyTlpAlgorithm

dtype = np.float32

def stress(pos2d, gt_DM, real_nodes_mask):
    pos2d = pos2d.astype(dtype)
    gt_DM = gt_DM.astype(dtype)

    pred_DM = pos2DM(pos2d, real_nodes_mask)

    mask = real_nodes_mask * real_nodes_mask.T
    gt_DM = gt_DM*mask
    pred_DM = pred_DM*mask

    gt_DM = gt_DM / np.max(gt_DM)
    pred_DM = pred_DM / np.max(pred_DM)
    

    dist_diff = np.square(gt_DM - pred_DM)
    alpha = 2
    epsilon = 1e-16
    zeros = np.zeros(gt_DM.shape)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in power")
        W = np.where(gt_DM < epsilon, zeros, gt_DM**-alpha) # raises a warning that can be ignored

    stress = W*dist_diff
    #lower is better
    return np.sum(stress)


def stress_normalized(pos2d, gt_DM, real_nodes_mask):
    global stress
    pos2d = pos2d.astype(dtype)
    gt_DM = gt_DM.astype(dtype)
    real_n_nodes = np.sum(real_nodes_mask)
    stress_res = stress(pos2d, gt_DM, real_nodes_mask)

    norm_stress = stress_res / (real_n_nodes**2)
    #lower is better
    return norm_stress


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    #angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def aspect_ratio(pos2d, gt_DM, real_nodes_mask):
    # gt_DM is not used but kept for the sake of the method signature
    # algo from GD² reversed AR = 1 - AR, lower is better
    real_n_nodes = int(np.sum(real_nodes_mask))
    AR = np.inf
    for k in range(real_n_nodes):
        theta = (2*math.pi*k) / real_n_nodes
        pos2d_rotated = np.zeros((real_n_nodes, 2))
        j = 0
        for i in range(pos2d.shape[0]):
            if(real_nodes_mask[i] == 1):
                pos2d_rotated[j] = rotate((0, 0), pos2d[i], theta)
                j += 1

        Xmin, Ymin = pos2d_rotated.min(axis=0)
        Xmax, Ymax = pos2d_rotated.max(axis=0)
        width = Xmax - Xmin
        height = Ymax - Ymin

        ratio = min(width, height) / max(width, height)
        AR = min(AR, ratio)
    #lower is better
    return 1 - AR


def sign(p, line):
    s = (p[0]-line[0][0])*(line[1][1]-line[0][1]) + (p[1]-line[0][1])*(line[0][0]-line[1][0])
    if(s > 0):
        return 1
    elif(s == 0):
        return 0
    else:
        return -1


def isSameSide(p1, p2, line):
    s1 = sign(p1, line)
    s2 = sign(p2, line)
    return s1 * s2


def hasCrossing(s1, s2, includeEndpoints=False):
    p1, p2 = s1
    p3, p4 = s2
    ss1 = isSameSide(p1, p2, [p3, p4])
    ss2 = isSameSide(p3, p4, [p1, p2])
    if (includeEndpoints):
        return ss1 <= 0 and ss2 <= 0
    else:
        return ss1 < 0 and ss2 < 0


def edges_crossing_number(pos2d, gt_DM, real_nodes_mask):
    # real_nodes_mask is not used but kept for the sake of the method signature
    pos2d = pos2d.astype(dtype)
    gt_DM = gt_DM.astype(dtype)

    edges = []
    N = gt_DM.shape[0]
    for i in range(N):
        for j in range(i+1, N, 1):
            if(gt_DM[i][j] == 1):
                edges.append((i, j))

    crossing_number = 0
    for i1 in range(len(edges)):
        for i2 in range(i1+1, len(edges), 1):
            e1 = edges[i1]
            e2 = edges[i2]
            p1, p2 = pos2d[e1[0]], pos2d[e1[1]]
            p3, p4 = pos2d[e2[0]], pos2d[e2[1]]
            crossing_number += hasCrossing((p1, p2), (p3, p4), includeEndpoints=False)
    #lower is better
    return crossing_number


def getAngle(c, p0, p1):
    c = np.array(c)
    p0 = np.array(p0)
    p1 = np.array(p1)

    p0c = np.linalg.norm(p0 - c)
    p1c = np.linalg.norm(p1 - c)
    p0p1 = np.linalg.norm(p0 - p1)
    return math.degrees(np.arccos((p1c**2+p0c**2-p0p1**2)/(2*p1c*p0c)))


def angular_resolution(pos2d, gt_DM, real_nodes_mask):
    min_angle = np.inf
    N = gt_DM.shape[0]
    # for each node
    for src in range(N):
        tgts = []
        for v in range(N):
            if(gt_DM[src][v] == 1):
                # get incident edges
                tgts.append(v)

        edges = [(src, tgt) for tgt in tgts]
        # for each pair of edges
        for i1 in range(len(edges)):
            for i2 in range(i1+1, len(edges), 1):
                e1 = edges[i1]
                e2 = edges[i2]
                assert e1[0] == e2[0]  # edges has the same src
                src = pos2d[e1[0]]
                p2 = pos2d[e1[1]]
                p3 = pos2d[e2[1]]
                # compute angle and keep smallest one
                angle = getAngle(src, p2, p3)
                min_angle = min(min_angle, angle)
    #higher  is better
    return min_angle # as degrees


def angular_resolution_normalized(pos2d, gt_DM, real_nodes_mask):
    AM = (gt_DM == 1).astype(int)
    min_angle = math.radians(angular_resolution(pos2d, gt_DM, real_nodes_mask))
    max_degree = np.max(np.sum(AM, axis=1))
    #anguler = 1 - angular, lower is better
    return 1 - min_angle / ((2*math.pi)/max_degree) # as radians


def autocorr_cluster_ambiguity(pos2d, gt_DM, real_nodes_mask, local_region_size_threshold=0.2):
    g, id_mapping = AM2tlp(gt_DM, real_nodes_mask)
    applyTlpAlgorithm(g, "MCL Clustering", "res_mcl")
    mcl_prop = g.getDoubleProperty("res_mcl")

    pos2d = pos2d.astype(dtype)
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)

    Cis=[]
    clusters = []
    for i in g.getNodes():
        if(mcl_prop.getNodeValue(i) not in clusters):
            clusters.append(mcl_prop.getNodeValue(i))
        summed_w = 0
        numerator = 0
        for j in g.getNodes():
            Dij = pred_DM[id_mapping[i.id]][id_mapping[j.id]]
            if(Dij < local_region_size_threshold):
                different_community = int((mcl_prop.getNodeValue(i) != mcl_prop.getNodeValue(j)))
                w = 1 - Dij
                summed_w += w
                numerator += w*different_community
        if(summed_w > 0):
            Ci = numerator / summed_w
            Cis.append(Ci)
    C = np.mean(Cis)
    #lower is better
    return C

def edges_length_uniformity(pos2d, gt_DM, real_nodes_mask, ideal_edge_length=None):
    #algo from GD2 ; ideal edge length is set to the mean length of edges
    AM = (gt_DM == 1).astype(float)
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)

    edges_dist_matrix = pred_DM*AM
    

    if(ideal_edge_length is None):
        ideal_edge_length = np.sum(edges_dist_matrix)  / np.sum(AM)

    uniformity = np.sum((((edges_dist_matrix) - ideal_edge_length*AM) / ideal_edge_length)**2)
    uniformity = np.sqrt(uniformity / np.sum(AM))
    #lower is better
    return uniformity

def neighborhood_preservation_gd2(pos2d, gt_DM, real_nodes_mask, k=3):
    # algo from gd2
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)
    pred_DM = shrinkDM(pred_DM, real_nodes_mask)
    gt_DM = shrinkDM(gt_DM, real_nodes_mask)
    AM = (gt_DM == 1).astype(int)

    real_n = int(np.sum(real_nodes_mask))

    pred_K = np.zeros(pred_DM.shape)
    pred_DM[pred_DM == 0] = np.inf
    for i in range(real_n):
        k_nearest_id = np.argsort(pred_DM[i])[:k]
        for j in k_nearest_id:
            pred_K[i][j] = 1    
    
    I = np.logical_and(AM, pred_K)
    U = np.logical_or(AM, pred_K)
    IoU = np.sum(I)/np.sum(U)
    return 1-IoU


def neighborhood_preservation_tsnet(pos2d, gt_DM, real_nodes_mask, r=2):
    # algo from tsnet paper
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)
    pred_DM = shrinkDM(pred_DM, real_nodes_mask)
    gt_DM = shrinkDM(gt_DM, real_nodes_mask)
    
    real_n = int(np.sum(real_nodes_mask))
    
    K_AM = np.zeros(gt_DM.shape)
    for i in range(real_n):
        for j in range(real_n):
            if(gt_DM[i][j] > 1-1e-8 and gt_DM[i][j] <= r):
                K_AM[i][j] = 1
    Ki = np.sum(K_AM, axis=1)
    pred_K_AM = np.zeros(pred_DM.shape)
    pred_DM[pred_DM == 0] = np.inf
    for i in range(real_n):
        ki = int(Ki[i])
        if(real_nodes_mask[i] == 1):
            k_nearest_id = np.argsort(pred_DM[i])[:ki]
            for j in k_nearest_id:
                if(not np.isinf(pred_DM[i][j])):
                    pred_K_AM[i][j] = 1
  
    I = np.logical_and(K_AM, pred_K_AM)
    U = np.logical_or(K_AM, pred_K_AM)
    IoU = np.sum(I) / np.sum(U)
    #higher is better, TODO: 1-IoU
    return 1-IoU

def get_nodes_diameter(resolution, X):
    return resolution / (X + X -1)

def nodes_resolution(pos2d, gt_DM, real_nodes_mask):
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)
    real_n = int(np.sum(real_nodes_mask))
    r = 1/np.sqrt(real_n) # or N_max ?

    overlap = min(1., np.min(pred_DM[pred_DM > 0]) / (r * np.max(pred_DM)))
    
    #reverse so that lower is better
    overlap = 1 - overlap
    return overlap

def euclidean_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1-p2)

def getNodePixels(nodeCenter, nodeRadius, resolution):
    pointsInXs = []
    pointsInYs = []
    xI = (nodeCenter[0] - nodeRadius, nodeCenter[0] + nodeRadius)
    yI = (nodeCenter[1] - nodeRadius, nodeCenter[1] + nodeRadius)
    for x in range(int(xI[0]),int(xI[1])+2, 1):
        if(x >= xI[0] and x <= xI[1] and x >= 0 and x <= resolution-1):
            #x is in
            for y in range(int(yI[0]),int(yI[1])+2, 1):
                if(y >= yI[0] and y <= yI[1] and y >= 0 and y <= resolution-1):
                    #y is in
                    if(euclidean_dist(nodeCenter, (x, y)) < nodeRadius):
                        pointsInXs.append(x)
                        pointsInYs.append(y)
    return pointsInXs, pointsInYs

def nodes_overlap(pos2d, gt_DM, real_nodes_mask):
    pred_DM = pos2DM(pos2d, real_nodes_mask)
    pred_DM = pred_DM / np.max(pred_DM)
    real_n = int(np.sum(real_nodes_mask))
    N_max = gt_DM.shape[0]

    R = 500
    positive_pos = pos2d
    nodes_radius = get_nodes_diameter(R, (real_n)/2) / 2
    real_node_sample_pos = positive_pos[np.where(real_nodes_mask == 1)[0][0]]
    positive_pos[np.where(real_nodes_mask == 0), :] = real_node_sample_pos

    if(np.min(positive_pos[:, 0]) < 0):
        positive_pos[:, 0] -= np.min(positive_pos[:, 0])
    if(np.min(positive_pos[:, 1]) < 0):
        positive_pos[:, 1] -= np.min(positive_pos[:, 1])

    positive_pos[:, 0] -= np.min(positive_pos[:, 0])
    positive_pos[:, 1] -= np.min(positive_pos[:, 1])    
    scaled_0_1 =MinMaxScaler().fit_transform(positive_pos.flatten().reshape((len(positive_pos.flatten()), 1))).reshape((N_max, 2))
    scaled_pos = scaled_0_1 * (R-2*nodes_radius) + nodes_radius
    drawing = np.zeros((R, R))
    for n in range(len(scaled_pos)):
        if(real_nodes_mask[n] == 1):
            p = scaled_pos[n]
            xs, ys = getNodePixels(p, nodes_radius, R)
            drawing[xs, ys] += 1

    drawing = np.rot90(drawing)
    theoretical_area = real_n * math.pi*nodes_radius**2
    overlapped = np.sum(drawing *(drawing>1))
    overlap_ratio = overlapped/theoretical_area
    return overlap_ratio

metrics = [
    "stress",
    "stress_normalized",
    "aspect_ratio",
    "edges_crossing_number",
    "angular_resolution",
    "angular_resolution_normalized",
    "autocorr_cluster_ambiguity",
    "edges_length_uniformity",
    "neighborhood_preservation_gd2",
    "neighborhood_preservation_tsnet",
    "nodes_overlap",
    "nodes_resolution"
]
