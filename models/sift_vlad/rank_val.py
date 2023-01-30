import time
import os
import numpy as np
import faiss
from pathlib import Path
from models.sift_vlad.utils.descriptor import get_des_vector_val
from models.sift_vlad.utils.kmeans_clustering import get_codebook
from models.sift_vlad.utils.vlad import get_vlad_base, get_pic_vlad


def ret_vlad_val(q_path_list, db_path_list, q_panorama_id, db_panorama_id, result_path, metric="cs", device='cpu', DESDIM=128):
    
    '''
    panorama_id : ID of panorama, specific location identified number including query and database
    metric : str, how to compare with vectors, ["l2", "cs"], l2 is Euclidean Distance, cs is Cosine Similarity
    device : str, if 'cpu', faiss search with cpu else, 'gpu' or 'cuda', anything.
    '''
    
    ##get all the descriptor vectors of the data set
    print("db_feature_num")
    all_des, image_des_len, _ = get_des_vector_val(db_path_list)

    ##get all the descriptor vectors of the query set
    print("q_feature_num")
    ret_des, ret_des_len, featCnt = get_des_vector_val(q_path_list)
    
    #Code book size auto setting
    COLUMNOFCODEBOOK = int(featCnt / 39) #39, faiss recommendation

    ##trainning the codebook
    NNlabel, codebook = get_codebook(all_des, COLUMNOFCODEBOOK) #DB 사이즈에 따라 CD 크기가 바뀌므로, 매번 뽑아줘야 한다.
    
    #get all vlad vectors of the db set
    vlad_base = get_vlad_base(image_des_len, NNlabel, all_des, codebook, COLUMNOFCODEBOOK)
    del all_des #save memory
    
    db_num = len(image_des_len)
    del image_des_len #save memory

    ##get all the vlad vectors of retrival set without pca dimensionality reduction
    cursor_ret = 0
    ret_vlad_list = []
    for i, eachretpic in enumerate(range(len(ret_des_len))): #각각 이미지 des 길이들 만큼만 pick! -> 이미지 1장 1장에 대해 retrieval 해야 하므로!
        pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
        ret_vlad = get_pic_vlad(pic, ret_des_len[eachretpic], codebook, COLUMNOFCODEBOOK) # COLUMNOFCODEBOOK #CD
        cursor_ret += ret_des_len[eachretpic] # next cursor
        ret_vlad_list.append(ret_vlad)
    del ret_des, ret_des_len #save memory

    ### for faiss ###
    ### kernel died because of doulbing memory by copying it
    vlad_base = np.array(vlad_base).reshape(len(vlad_base), -1) #(개수, 차원) ex) K=5500 (3369, 704000)
    vlad_base = vlad_base.astype(np.float32)
    
    ret_vlad_set = np.array(ret_vlad_list).reshape(len(ret_vlad_list), -1) #(개수, 차원) ex) K=5500, (100, 704,000)
    ret_vlad_set = ret_vlad_set.astype(np.float32)
    
    ## faiss similarity search ##
    faiss.normalize_L2(vlad_base) # 자신의 절댓값 크기(sqrt(self**2))로 나눠준다.
    faiss.normalize_L2(ret_vlad_set)
    
    dim = COLUMNOFCODEBOOK * DESDIM # vlad_dim: K*D
    if metric == "l2":  # Norm L2 Dist
        cpu_index = faiss.IndexFlatL2(dim)
        
    else : # metric == "cs", cosine similarity
        cpu_index = faiss.IndexFlatIP(dim)
        
    if device == 'cpu':
        cpu_index.add(vlad_base)
        del vlad_base #save memory

        D, I = cpu_index.search(ret_vlad_set, cpu_index.ntotal) # 전체 쿼리(병렬) 몇 등까지 볼거야? 전체 다!
        
    else: #'cuda', 'gpu'
        
        #Clone from CPU to GPU
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(cpu_index, co=co) # build the index
        index.add(vlad_base)
        del vlad_base #save memory
        
        D, I = index.search(ret_vlad_set, index.ntotal) # 전체 쿼리(병렬) 몇 등까지 볼거야? 전체 다!
    
    q_num = ret_vlad_set.shape[0]
    del ret_vlad_set #save memory
    
    #out result dic
    result_dic = {}
    
    for q_idx in range(q_num): # query index
        eval_list = []
        for db_idx in range(db_num): # db index
            eval_list.append((I[q_idx,db_idx], D[q_idx,db_idx]))
        result_dic[q_idx] = eval_list

    # save top1 pair .txt
    os.makedirs(result_path + '/eval/sift_best_pair/', exist_ok=True)
    out_file = result_path + '/eval/sift_best_pair/' + f"pair_{q_panorama_id}-{db_panorama_id}_sift" + ".txt"
    
    # if repeated, the lines is accumulated
    if os.path.isfile(out_file):
        os.remove(out_file)
    
    f_out = open(out_file, 'w')
    for q_idx in list(result_dic.keys()):
        
        query_name = Path(q_path_list[q_idx]).stem
        #### L2 Distance/ Cos_Sim ####
        if metric == "l2":  # Norm L2 Dist
            rank_list = sorted(result_dic[q_idx], key = lambda x: x[1]) # Norm l2 dist
        else : # cosine similarity
            rank_list = sorted(result_dic[q_idx], key = lambda x: -x[1]) #cos sim
        ##############################
        for (db_idx, score)  in rank_list:
            
            db_name = Path(db_path_list[db_idx]).stem
            
            f_out.write(query_name + '.jpg,' + db_name + '.jpg,' + str(score) + '\n')
            f_out.flush()
    f_out.close()
    #done
    print(f"\nSaving...\n{out_file}\nDone.")