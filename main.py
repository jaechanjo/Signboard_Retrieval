import cv2
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from pathlib import Path
import argparse
import faiss

from utils.crop import crop_get
from models.sift_vlad.rank import ret_vlad
from models.vit.main import main as ret_vit
from utils.merge import merge_topk
from utils.visualization import dic2visualization


def main(q_img_path, db_img_path,\
         result_path='./data/result/', topk=1, match_weight='1/4', method='vit', algo='max', device='cuda', batch_size=64, num_workers=0):
    
    #crop panorama query & db
    q_crop_list, db_crop_list, q_panorama_id, db_panorama_id = crop_get(q_img_path, db_img_path)
    
    #multi processing
    proc1 = Process(target=ret_vlad, args=(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, result_path, "cs", device))
    proc2 = Process(target=ret_vit, args=(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, f"{result_path}/vit_best_pair/", batch_size,\
                                     num_workers, device))
    
    proc1.start(); proc2.start()
    proc1.join(); proc2.join()

    #merge_topk
    #str2float
    match_weight = eval(match_weight)
    result_dict, result_json = merge_topk(result_path, q_img_path, db_img_path, q_panorama_id, db_panorama_id, topk, match_weight, method, algo)
    
    return result_dict, result_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--q_img_path', nargs='?', type=str, help='query panorama image path')
    parser.add_argument('--db_img_path', nargs='?', type=str, help='db panorama image path')
    parser.add_argument('--result_path', type=str, default='./data/result/')
    parser.add_argument('--topk', type=int, default=1, help='the number of matching candidates')
    parser.add_argument('--match_weight', type=str, default='1/4', \
                        help='threshold of whether matched or not')
    parser.add_argument('--method', type=str, default='vit', \
                        help="module, ['vit', 'sift', 'vit_sift', 'sift_vit']")
    parser.add_argument('--algo', type=str, default='max', \
                        help="matching algorithm, ['max', 'erase']")
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--batch_size', type=int, default=64, \
                        help='the batch size extracting vit feature') 
    parser.add_argument('--num_workers', type=int, default=0)
    
    opt = parser.parse_args()
    
    ### signboard retrieval ###
    result_dict, result_json = main(opt.q_img_path, opt.db_img_path, \
         opt.result_path, opt.topk, opt.match_weight, opt.method, opt.algo, \
         opt.device, opt.batch_size, opt.num_workers)
    
    ###### print result ###
    # print(f"{result_dict}\n\n")
    # print(result_json)
    
    ### visualization ###
    dic2visualization(opt.q_img_path, opt.db_img_path,\
                      result_dict, f"{opt.result_path}/visualization/", opt.method)
    
    
