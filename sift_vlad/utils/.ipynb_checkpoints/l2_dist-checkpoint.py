import numpy as np

def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors (norm_l2)
    '''
    return np.linalg.norm(vec1 - vec2)