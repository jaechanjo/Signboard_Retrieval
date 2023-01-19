import time
import numpy as np
import os 
import natsort
from pathlib import Path

def merge_topk(match_score_dir, q_crop_dir, db_crop_dir, panorama_id, topk, val=False, score_thres=1.2/1.8):
    
    '''
    val: bool, whether evaluate result by mAP@topk or not
    '''
    
    #merge
    method = ['vit'] #['sift',  # , 'ocr']
    merge_dict = {}
    
    for m in method:
        # try:
        with open(f"{match_score_dir}/{m}_best_pair/pair_{panorama_id}_{m}.txt", "r") as f:
            for line in f.readlines():
                data = line.strip().split(',')
                img1 = data[0].split('.')[0]
                img2 = data[1].split('.')[0]
                score = match_score(q_crop_path=q_crop_dir+f"/{panorama_id}/{data[0]}",\
                                    db_crop_path=db_crop_dir+f"/{panorama_id}/{data[1]}",\
                                    score_thres=score_thres)
                if img1+'-'+img2 not in merge_dict: #3가지 방법에서 특정 q_db pair가 중복된다면, 점수를 합산!
                    merge_dict[img1+'-'+img2] = score
                else:
                    merge_dict[img1+'-'+img2] += score
        # except:
        #     print(f"method {m} error")
            
    #change form
    result_dict = {}
    
    for key in merge_dict:
        imgs = key.split('-')
        q = imgs[0]
        db = imgs[1]
        del imgs
        score = merge_dict[key]
        
        #특정 query에 대해, 모든 ref를 list에 모아주기
        if q not in result_dict: #이 query가 처음 나왔을 때
            result_dict[q] = [(db, int(score))]
        else: #query가 중복될 경우
            db_list = result_dict[q]
            db_list.append((db, int(score)))
            result_dict[q] = db_list
            
    #topk
    result_topk = {}
    
    for q in result_dict.keys():
    
        topk_list = sorted(result_dict[q], key=lambda x: -x[1])[:topk] #match_score descending
        topk_db = list(map(lambda x: x[0], topk_list)) #only db name
        result_topk[q] = topk_db
        
    #save result dict
    # os.makedirs(save_path, exist_ok=True)
    # with open(save_path + f"pair_{panorama_id}_top{topk}_final" + ".pkl", 'wb') as fp:
    #     pickle.dump(result_topk, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    #eval mAP@topk
    #ex) query_name : '0_100_False'/ db_name : '2_100_False'
    if val:
        
        #panorama mAP : topk 안에 들면 +1/ 아니면 0 -> 파노라마 당 ap 
        pap = 0
        #crop mAP : 간판 크롭 1개 당 ap
        ap = 0
        for q_name in list(result_topk.keys()):
            #query_label
            q_lbl = int(q_name.split('_')[1])

            for rank, db_name in enumerate(result_topk[q_name], start=1):
                #db_label
                db_lbl = int(db_name.split('_')[1])

                if (q_lbl == db_lbl) & (q_lbl != 100): #& ('True' not in q_name) : ##Count #100 not matched / + except change
                    #panorama mAP
                    pap += 1
                    #crop mAP
                    qap = 1/rank
                    ap += qap
        
        #crop mAP
        matched_cnt = len([q_name for q_name in result_topk.keys() if ('100' not in q_name)]) #& ('True' not in q)]) ##Divide #except unmatched '100' / + except change
        
        #panorama mAP
        try:
            pap/=len([q_name for q_name in result_topk.keys() if ('100' not in q_name)]) #& ('True' not in q)]) ##Divide #except unmatched '100' / + except change
            print(f"panorama AP@{topk} is {pap}") #result_topk 자체가 filter
                
        except:
            if matched_cnt == 0:
                print(f"there is no gt, pass!") #result_topk 자체가 filter
            else:
                print("stop! - what happen now?")
            
    return result_topk, pap, (ap, matched_cnt)

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


