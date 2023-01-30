import time
import numpy as np
import os 
import natsort
from pathlib import Path

def new_mAP(args, I, q_tmp_img_dir_idx, db_tmp_img_dir_idx):
    q_imgs_path_list = []
    new_q_imgs_path_list = []
    db_imgs_path_list = []
    ### 쿼리 이미지 인덱싱 ##
    
    for root, dirs, files in os.walk(q_tmp_img_dir_idx):
        for file in files:
            q_imgs_path_list.append(os.path.join(q_tmp_img_dir_idx, file))
    q_imgs_path_list = natsort.natsorted(set(q_imgs_path_list))
    #print()
    ### 디비 이미지 인덱싱 ##
    for root, dirs, files in os.walk(db_tmp_img_dir_idx):
        for file in files:
            db_imgs_path_list.append(os.path.join(db_tmp_img_dir_idx, file))
    db_imgs_path_list = natsort.natsorted(set(db_imgs_path_list))

    result_dic = {}
    #eval_list.append((I[q_idx,db_idx], D[q_idx,db_idx])) ## D, I
    print(q_imgs_path_list)
    pap = []

    ap=0

    for q_idx in range(len(q_imgs_path_list)): # query index
        q_name = os.path.basename(q_imgs_path_list[q_idx]) ##
        q_lbl = q_name.split('_')[1]
        # print(f"q_imgs_path_list is {q_imgs_path_list}")
        # print(f"--"*10)
        # print(f"q_lbl is {q_lbl}")

        eval_list = []
        for i, db_idx in enumerate(range(I.shape[1])): # db index
            if i+1 <= args.top_k: #map@topk
                db_name = Path(db_imgs_path_list[db_idx]).stem ##
                db_lbl = db_name.split('_')[1]
                #print(f"db_lbl is {db_lbl}")
                #print(f"topk is {args.top_k}")
                if (q_lbl == db_lbl) & ('100' != q_lbl):
                    ap += 1
        #print(f"ap is {ap}")
    #     result_dic[q_idx] = eval_list
    # #panorama ap
    # ap/=(len([q_path for q_path in q_imgs_path_list if (Path(q_path).stem.split('_')[1] != 100])) # matched query num

    #     # pap.append(ap)
    
    # # pap/=(LEN(파노라마 전체 개수))


    # # get_result_time
    # print("get_result_dic", time.perf_counter()-start_rd)


    # final_map = np.mean(pap)

#return q_imgs_path_list

def calculate_ap(idx, gt):    
    ap,rank=0.,None    
    for n,i in enumerate(idx,start=1):
        if i==gt:
            ap=1/n
            rank=n
            break
    return ap,rank

def calculate_mAP(top_k, I):
    rank_sum = 0
    sum_AP=0.    
    start = time.time()
    for i in range(I.shape[0]):        
        rank = np.where(I[i] == i)
        print(i, rank)
        rank_sum += rank[0][0]
        idx=I[i]
        if top_k:
            idx=I[i,:top_k]
        ap,r=calculate_ap(idx,i)
        sum_AP+=ap
        print(f'[Query {i+1}] Rank: {r}, AP: {ap:.4f}, mAP: {sum_AP/(i+1)}')
    mAP = sum_AP/(i+1)
    print(f'{time.time() - start:.2f} sec')

    return mAP, rank_sum

# #쿼리 인덱스가 디비 인덱스 넘으면 rank 에러 생김 반드시 해결할것
# def new_calculate_mAP(top_k, D, I):
#     rank_sum = 0
#     sum_AP=0.    
#     print(D)
#     print('=='*10)
#     print(I)
#     start = time.time()
#     for i in range(I.shape[0]):        
#         #print(I.shape)
#         # if len(I.shape[0]) > len(I.shape[1]) #쿼리 길이 > 디비 길이 
        
#         # else:
#         rank = np.where(I[i] == i) #[ 7  0  4  9 11 12 10  5 14 16 15  3  8  2 13  6  1]
#         print(i, I[i])
#         rank_sum += rank[0][0]
#         idx=I[i]
#         if top_k:
#             idx=I[i,:top_k]
#         ap,r=calculate_ap(idx,i)
#         sum_AP+=ap
#         print(f'[Query {i+1}] Rank: {r}, AP: {ap:.4f}, mAP: {sum_AP/(i+1)}')
#     mAP = sum_AP/(i+1)
#     print(f'{time.time() - start:.2f} sec')

#     return mAP, rank_sum


