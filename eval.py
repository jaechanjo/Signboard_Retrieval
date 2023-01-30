import cv2
import time
import json
import os
import numpy as np
import natsort
import matplotlib.pyplot as plt
from multiprocessing import Process
from pathlib import Path
import argparse
import faiss

from utils.parsing import parsing
from models.vit.main import main_val as ret_vit_val
from models.sift_vlad.rank_val import ret_vlad_val as ret_vlad_val
from utils.merge_val import merge_topk_val


def evaluation(query_dir, db_dir,\
               result_path='./data/result/', topk=1, match_weight='1/4', method='vit', algo='max', device='cuda', batch_size=64, num_workers=0):
    
    #crop & parsing label from yolov7 detector
    query_val_dir, db_val_dir = parsing(query_dir, db_dir)
    
    #query& db path dictionary
    pnorm_db_dic = {} #panorama db 
    for db_idx in os.listdir(db_val_dir):
    
        db_pnorm_dir = db_val_dir + db_idx + '/'
        db_path_list = natsort.natsorted([db_pnorm_dir + db_name for db_name in os.listdir(db_pnorm_dir) if 'jpg' in db_name or 'png' in db_name])
        pnorm_db_dic[db_idx] = db_path_list

    pnorm_q_dic = {} #panorama query
    for q_idx in os.listdir(query_val_dir):
    
        q_pnorm_dir = query_val_dir + q_idx + '/'
        q_path_list =natsort.natsorted( [q_pnorm_dir + q_name for q_name in os.listdir(q_pnorm_dir) if 'jpg' in q_name or 'png' in q_name])
        pnorm_q_dic[q_idx] = q_path_list

    #multi processing(SIFT/VIT) - Matching& Rank
    for i, p_id in enumerate(pnorm_q_dic.keys()): #p_id : panorama_id

        #process
        print(f"Panoram_ID : {p_id} - {i+1}/{len(pnorm_q_dic)}")

        try: # query ID == db ID

            proc1 = Process(target=ret_vlad_val, args=(pnorm_q_dic[p_id], pnorm_db_dic[p_id], p_id, p_id, result_path, "cs", device))
            proc2 = Process(target=ret_vit_val, args=(pnorm_q_dic[p_id], pnorm_db_dic[p_id], p_id, p_id, result_path, batch_size, num_workers, device))

            proc1.start(); proc2.start()
            proc1.join(); proc2.join()

        except: # if query ID not in db ID
            print("ID matching error")
            print(f"query ID : {p_id}")
    
    #variables
    match_weight = eval(match_weight)  # str2float
    final_result_topk = {} # result

    #score
    matched_score = []
    unmatched_score = []

    #panorama mAP
    pap_list = []
    
    #crop mAP
    cap = 0
    matched_cnt = 0

    # run merge_val
    for i, p_id in enumerate(pnorm_q_dic.keys()):
        
        #process
        print(f"Panoram_ID : {p_id} - {i+1}/{len(pnorm_q_dic)}")

        try:# query ID == db ID
            result_topk, pap, cap_tuple, score_tuple = merge_topk_val(result_path, p_id, p_id, \
                                                                      topk, match_weight, method, algo)
            print(result_topk)
            final_result_topk[p_id] = result_topk #[(db_name, score), ..., (db_name, score)]

            #score thres?
            matched_score += score_tuple[0]
            unmatched_score += score_tuple[1]

            #panorama mAP
            pap_list.append(pap)
            #crop mAP == recall
            cap += cap_tuple[0]
            matched_cnt += cap_tuple[1]

        except: # panorama id가 없을 때
            raise Exception(f"panorama_id {p_id} error")
    
    print("\Merge...\nDone.")
    ################ Recall& Precision #####################

    # magic line for distinguish 'matched' from 'unmatched'
    match_thres = match_weight*np.mean(matched_score) + (1-match_weight)*np.mean(unmatched_score)

    # final result dict
    final_dict = {}

    # TP : True Positive
    # FP : False Positive
    tp = 0
    fp = 0 

    for p_id in list(final_result_topk.keys()): #p_id : panorama_id

        thres_result = {}

        for q_name in final_result_topk[p_id].keys():
            #query_label
            q_lbl = int(q_name.split('_')[1])

            thres_list = []

            for rank, (db_name, score) in enumerate(final_result_topk[p_id][q_name], start=1): #only db_name
                #db_label
                db_lbl = int(db_name.split('_')[1])

                if score > match_thres: # positve : predict top 1 db

                    if (q_lbl == db_lbl) & (q_lbl != 100):
                        tp += 1
                    else:
                        fp += 1

                    thres_list.append(db_name)    
                else: # negative : predict empty list []
                    pass
            thres_result[q_name] = thres_list

        final_dict[p_id] = thres_result

    print(f'macro_mAP@{topk} : {round(np.mean(pap_list),2)}')
    print(f'micro_mAP@{topk} : {round(cap/matched_cnt,2)}')
    print(f"recall@1:{round(tp/matched_cnt,2)}") 
    print(f"precision@1:{round(tp/(tp+fp),2)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--query_dir', nargs='?', type=str, help='query panorama image directory')
    parser.add_argument('--db_dir', nargs='?', type=str, help='db panorama image directory')
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
    
    ### evaluation
    evaluation(opt.query_dir, opt.db_dir, \
               opt.result_path, opt.topk, opt.match_weight, opt.method, opt.algo, \
               opt.device, opt.batch_size, opt.num_workers)   
