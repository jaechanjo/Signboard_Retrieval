import argparse
import time
import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt
import natsort
import os 
import itertools
import cv2
import csv
import pickle
from pathlib import Path

from models.vit.utils import metric

def search(query_feature, reference_feature):
    ### query feature extracting ### 
    query_feat = query_feature.numpy()
    faiss.normalize_L2(query_feat) # feature normalize_L2 for compare
    
    ### reference feature extracting ###
    reference_feat = reference_feature.numpy()
    faiss.normalize_L2(reference_feat) # feature normalize_L2 for compare
    
    ### image feature similarity comparing ###  
    index = faiss.IndexFlatIP(reference_feat.shape[1])
    
    index.add(reference_feat)
    D, I = index.search(query_feat, reference_feat.shape[0])

    return D, I

    
#evaluation     
def make_result_val(D, I, q_panorama_id, db_panorama_id, result_path, q_path_list, db_path_list):
    
    #make directory of eval
    os.makedirs(result_path + "/eval/vit_best_pair/", exist_ok=True)
    res_txt_path = result_path + "/eval/vit_best_pair/" + f"pair_{q_panorama_id}-{db_panorama_id}_vit.txt"
    
    # if repeated, the lines is accumulated
    if os.path.isfile(res_txt_path):
        os.remove(res_txt_path)
    
    with open(res_txt_path, 'a', encoding='utf-8') as f:
        for n, q_idx in enumerate(range(I.shape[0])):
            
            query_name = Path(q_path_list[q_idx]).stem
            
            for db_idx in range(I.shape[1]):
                
                db_name = Path(db_path_list[I[q_idx][db_idx]]).stem
                
                txt = str(query_name)+'.jpg,'+str(db_name)+'.jpg,'+str(D[q_idx][db_idx])
                f.write(txt+'\n')
                f.flush()
    print(f"\nSaving...\n{res_txt_path}\nDone.")

    
def main(query_feature, reference_feature, q_panorama_id, db_panorama_id, result_path, q_path_list, db_path_list):

    D, I = search(query_feature, reference_feature)
    make_result_val(D, I, q_panorama_id, db_panorama_id, result_path, q_path_list, db_path_list)

if __name__ == '__main__':
    # feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    query_path = "/home/signboard_retrieval/features/q_crop_val/0/0_token_cls.pth"
    reference_path = "/home/signboard_retrieval/features/db_crop_val/0/0_token_cls.pth"
    
    main(query_feature, reference_feature, panorama_id=0, result_path=args.result_path)
