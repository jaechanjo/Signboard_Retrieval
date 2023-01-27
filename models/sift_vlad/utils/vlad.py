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
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM]).astype(np.float32) # VLAD vector: K * D 차원
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes] #차원 별 각각 중심으로부터 거리 (lacal feature 값 - NN 중심 값)
        cursor += eachImage # cursor 이동
    
        vlad_norm = vlad.copy()
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2) # 0에서 1사이로 L2 Norm
        vlad_base.append(vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)) # vlad_norm : 이미지 1장 -> vlad_base : DB 모든 이미지 M장 = (K*D)*M
    
    print("get_db_vlad_feat")
    
    return vlad_base #VLAD Base Vector


# query에 대해서 ON_LINE에서 진행되는 과정
def get_pic_vlad(pic, des_size, codebook, COLUMNOFCODEBOOK, DESDIM=128): # query의 vlad vector 만들기 by db의 codebook으로!
    '''
    Description: get the vlad vector of each image
    '''
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = 1000000000.0
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i]) # 각각 local feature가 중심으로부터 떨어진 거리
            if dist < min_dist: # 최근접 중심
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind] # 최근접 중심 + 떨어진 거리 : Vlad 정의
    
    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2) #L2 Norm
    vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)
    
    return vlad_norm

