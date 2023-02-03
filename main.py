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

from config.params import *
from utils.crop import crop_get
from models.sift_vlad.rank import ret_vlad
from models.vit.main import main as ret_vit
from utils.merge import merge_topk
from utils.visualization import dic2visualization

class SIFT_VIT:
    def __init__(self):
        #params
        self.result_path = result_path
        self.topk = topk
        self.match_weight = eval(match_weight)
        self.method = method
        self.algo = algo
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def inference(self, query_path, db_path, query_json, db_json):
    
        #crop panorama query & db
        q_crop_list, db_crop_list, q_panorama_id, db_panorama_id = crop_get(query_path, db_path, query_json, db_json)
        
        #multiprocessing
        proc1 = Process(target=ret_vlad, args=(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, self.result_path, "cs", 'cpu'))
        proc2 = Process(target=ret_vit, args=(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, self.result_path, self.batch_size, self.num_workers, 'cuda'))
        
        #multi processing
        proc1.start(); proc2.start()
        proc1.join(); proc2.join()

        #merge_topk
        #str2float
        result_dict, result_json = merge_topk(self.result_path, query_path, db_path, q_panorama_id, db_panorama_id, self.topk, self.match_weight, self.method, self.algo)

        return result_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--query_path', nargs='?', type=str, help='query panorama image path')
    parser.add_argument('--db_path', nargs='?', type=str, help='db panorama image path')
    parser.add_argument('--query_json_path', nargs='?', type=str, help='query label json path')
    parser.add_argument('--db_json_path', nargs='?', type=str, help='db label json path')
    parser.add_argument('--result_path', type=str, default='./data/result/')
    parser.add_argument('-vis', '--visualize', action='store_true', help='Whether to save the resulting image')
    parser.add_argument('--topk', type=int, default=1, help='the number of matching candidates')
    parser.add_argument('--match_weight', type=str, default='1/4', \
                        help='threshold of whether matched or not')
    parser.add_argument('--method', type=str, default='vit', \
                        help="module, ['vit', 'sift', 'vit_sift', 'sift_vit']")
    parser.add_argument('--algo', type=str, default='max', \
                        help="matching algorithm, ['max', 'erase']")
    parser.add_argument('--batch_size', type=int, default=64, \
                        help='the batch size extracting vit feature') 
    parser.add_argument('--num_workers', type=int, default=0)
    
    opt = parser.parse_args()
    
    #load json
    with open(opt.query_json_path, 'r') as qj:
        query_det = json.load(qj)
    with open(opt.db_json_path, 'r') as dbj:
        db_det = json.load(dbj)
    
    #load model
    model = SIFT_VIT()
    
    #inference
    result_json = model.inference(opt.query_path, opt.db_path, query_det, db_det)
    
    ###### print result ###
    # print(f"{result_dict}\n\n")
    # print(result_json)
    
    ### visualization ###
    if opt.visualize:
        dic2visualization(opt.query_path, opt.db_path,\
                          result_dict, f"{opt.result_path}/visualization/", opt.method)