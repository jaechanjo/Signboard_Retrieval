import numpy as np
import faiss

def get_cluster_center(des_set, K):
    '''
    Description: cluter using a default setting
    Input: des_set - cluster data
                 K - the number of cluster center
    Output: laber  - a np array of the nearest center for each cluster data
            center - a np array of the K cluster center
    '''
    des_set = np.float32(des_set) # for kmeans 

    # Setup
    kmeans = faiss.Kmeans(d=128, k=K, niter=20, verbose=True, gpu=True)
    
    # Run clustering
    kmeans.train(des_set)
    
    # Centroids after clustering
    center = kmeans.centroids
    
    # The assignment for each vector.
    _, label = kmeans.index.search(des_set, 1)  # Need to run NN search again
    
    return label, center


def get_codebook(all_des, K):
    '''
    Description: train the codebook from all of the descriptors
    Input: all_des - training data for the codebook
                 K - the column of the codebook
    '''
    label, center = get_cluster_center(all_des, K)
    return label, center #center -> codebook