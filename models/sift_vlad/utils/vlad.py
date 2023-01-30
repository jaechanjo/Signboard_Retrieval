import cv2
import numpy as np
from models.sift_vlad.utils.l2_dist import cal_vec_dist

def get_vlad_base(img_des_len, NNlabel, all_des, codebook, COLUMNOFCODEBOOK, DESDIM=128):
    '''
    Description: get all images vlad vector 
    '''
    cursor = 0
    vlad_base = []
    for i, eachImage in enumerate(img_des_len):
        
        descrips = all_des[cursor : cursor + eachImage]
        centriods_id = NNlabel[cursor : cursor + eachImage]
        centriods = codebook[centriods_id]
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM]).astype(np.float32) # VLAD vector: K * D dim
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
        cursor += eachImage # move cursor
    
        vlad_norm = vlad.copy()
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
        vlad_base.append(vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)) 
    
    print("get_db_vlad_feat")
    
    return vlad_base #VLAD Base Vector


def get_pic_vlad(pic, des_size, codebook, COLUMNOFCODEBOOK, DESDIM=128):
    '''
    Description: get the vlad vector of each image
    '''
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = 1000000000.0
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i]) 
            if dist < min_dist: 
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind] 
    
    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2) #L2 Norm
    vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)
    
    return vlad_norm

