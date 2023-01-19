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
    
##########################[CPU-SLOW OPENCV KMEANS]##########################
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    
    # ret, label, center = cv2.kmeans(des_set, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
#########################################################################

##########################[GPU-FAST FAISS KMEANS]##########################
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