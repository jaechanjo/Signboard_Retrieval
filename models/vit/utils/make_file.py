import os 
import argparse
import csv
import pickle
import natsort


def find_files(args):
    q_idx_path_list = []
    db_idx_path_list = []

    # return query image file path
    for root, dirs, files in os.walk(args.q_img_path):
        for file in files:
            if file.endswith('.jpg'):
                q_idx_path_list.append(os.path.join(root))
    q_idx_path_list = natsort.natsorted(set(q_idx_path_list))
    
    # return ref image file path
    for root, dirs, files in os.walk(args.db_img_path):
        for file in files:
            if file.endswith('.jpg'):
                db_idx_path_list.append(os.path.join(root))
    db_idx_path_list = natsort.natsorted(set(db_idx_path_list))

    return q_idx_path_list, db_idx_path_list



def get_rank(D, I, args, query_path):
    f = open(f'{args.result_path}/rank.txt', 'a')
    
    # get mAP and rank 
    start = time.time()
    mAP, rank_sum = metric.calculate_mAP(args.top_k, I)
    avg_rank = rank_sum / I.shape[0]
    print(f'[Total] mAP: {mAP}')

    # save rank.txt
    rank_per_model = args.model + '_' + str(args.image_size) + '_' + str(args.image_mode) + '_' + query_path.split('.')[-2][6:] + '_top_' + str(args.top_k)
    mAP_per_model = args.model + '_' + str(args.image_size) + '_' + str(args.image_mode) + '_' + query_path.split('.')[-2][6:] + '_top_' + str(args.top_k)
    print(f'{rank_per_model},\trank: {rank_sum / I.shape[0]: .6f},\tmAP: {mAP/(I.shape[0])}', file=f)
    f.close()
    print(f'{rank_per_model}: {rank_sum / I.shape[0]: .6f}')
    
    return mAP, avg_rank, query_path

def make_csv(args, mAP, avg_rank, query_path, reference_feat):
    #row_header = img_size,latio,model,layer,top_k,mAP,avg_rank,feat_dims
    f = open(f'{args.result_path}/result.csv', 'a', encoding='utf-8')
    f.write(str(args.image_size)+','+args.image_mode+','+args.model+','+query_path.split('.')[-2][6:]+','+'top@'+str(args.top_k)+','+str(mAP)+','+str(avg_rank)+','+str(reference_feat.shape[1])+'\n')
    f.close()

def make_dict(args, D, I):
    ranking_dict={}
    for n, i in enumerate(range(I.shape[0])):
        q_idx=i+1 #key 
        rank = np.where(I[i] == i)
        rank_val = rank[0][0]+1 #value
        distance = D[i][rank] #value
        distance_val = distance[0]
        
        print(f"==="*10)
        print(f"D is \n{D}")
        print(f"I is \n{I}\n")
        print(f"q_idx is {q_idx}")
        print(f"rank is {rank[0][0]+1}")
        print(f"distance is {distance[0]}")
        print(f"==="*10)
        ranking_dict[q_idx]=(rank_val, distance_val)

    print(f"ranking_dict is {ranking_dict}")

    with open(f'{args.result_path}/beitv2_large_patch16_224_in22k_224_original_fc_layer.p', 'wb') as f:
        pickle.dump(ranking_dict, f)

    with open(f'{args.result_path}/beitv2_large_patch16_224_in22k_224_original_fc_layer.p', 'rb') as f:
        data = pickle.load(f)
    print(f"data is {data}")
