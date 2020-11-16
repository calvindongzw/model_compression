import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

from heapq import heappush, heappop, heapify
from collections import defaultdict, namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: encoding map and frequency map for the current weight layer.
    """
    
    frequency = defaultdict(int)
    for i in weight.flatten():
        frequency[i] += 1
        
    Node = namedtuple('Node', 'f v l r')
    Node.__lt__ = lambda x, y: x.f < y.f
    heap = [Node(f, v, None, None) for v, f in frequency.items()]
    heapify(heap)
        
    while(len(heap) > 1):
        n1 = heappop(heap)
        n2 = heappop(heap)
        merged = Node(n1.f + n2.f, None, n1, n2)
        heappush(heap, merged)
        
    encodings = {}
    
    def create_code(node, code):
        if node is None:
            return
        if node.v is not None:
            encodings[node.v] = code
            return
        create_code(node.l, code + '0')
        create_code(node.r, code + '1')
    
    root = heappop(heap)
    create_code(root, '')   
        
    return encodings, frequency

def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param encodings: encoding map of the current layer w.r.t. weight (centriod) values.
    :param frequency: frequency map of the current layer w.r.t. weight (centriod) values.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    huffman_total = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            huffman_total += huffman_avg_bits
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            huffman_total += huffman_avg_bits
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map, huffman_total