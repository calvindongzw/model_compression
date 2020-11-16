import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

from scipy.sparse import csc_matrix, csr_matrix
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def linear_init(weight, bits):
    init_cluster = np.linspace(np.min(weight), np.max(weight), num=2**bits)
    return init_cluster

def random_init(weight, bits):
    init_cluster = np.random.choice(weight.flatten(), size=2**bits)
    return init_cluster

def dense_init(weight, bits):
    d, _ = np.histogram(weight.flatten(), bins=len(weight.flatten()), density=True)
    prob = d * np.diff(_)
    init_cluster = np.random.choice(weight.flatten(), 2**bits, p=prob)
    return init_cluster

def _quantize_layer(weight, init, bits=8):
    """
    :param weight: A numpy array of any shape.
    :param bits: quantization bits for weight sharing.
    :return quantized weights and centriods.
    """

    if init == 'linear':
        init = linear_init(weight, bits) 
    if init == 'random':
        init = random_init(weight, bits)  
    if init == 'dense':
        init = dense_init(weight, bits)
    
    shape = weight.shape
    sp_weight = csr_matrix(weight.reshape(-1,1))

    kmeans = KMeans(n_clusters=2**bits, init=init.reshape(-1,1), n_init=1)
    kmeans.fit(sp_weight.data.reshape(-1,1))
    centers_ = kmeans.cluster_centers_
    sp_weight_update = centers_[kmeans.labels_]
    sp_weight.data = sp_weight_update
    new_weight = sp_weight.toarray().reshape(shape)
            
    return new_weight, centers_

def quantize_whole_model(net, init, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, init, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.conv.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            #print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, init, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.linear.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            #print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

