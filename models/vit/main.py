import argparse
import os
import natsort
import pickle5 as pickle

import models.vit.feature_extractor as feature_extractor
import models.vit.feature_extractor_val as feature_extractor_val
import models.vit.rank as rank
import models.vit.rank_val as rank_val

from models.vit.utils.make_file import find_files #__init__.py 로 다시 수정해보기 

def main(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, result_path, batch_size=64, num_workers=0, device='cuda'):
    
    query_feature, reference_feature = feature_extractor.main(q_crop_list, db_crop_list, batch_size, num_workers, device)
    rank.main(query_feature, reference_feature, q_panorama_id, db_panorama_id, result_path)
    
#evaluation
def main_val(q_path_list, db_path_list, q_panorama_id, db_panorama_id, result_path, batch_size=64, num_workers=0, device='cuda'):
    
    query_feature, reference_feature = feature_extractor_val.main(q_path_list, db_path_list, batch_size, num_workers, device)
    rank_val.main(query_feature, reference_feature, q_panorama_id, db_panorama_id, result_path, q_path_list, db_path_list)

if __name__ == '__main__':
    # feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature') 
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')   
    args = parser.parse_args()

    ###나중에 삭제##
    with open('/home/signboard_retrieval/pkl/db_panorama_list.pkl', 'rb') as fp: #len은 동일
        db_panorama_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/q_panorama_list.pkl', 'rb') as fp: #len은 동일
        q_panorama_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/db_crop_list.pkl', 'rb') as fp: #len은 동일
        db_crop_list = pickle.load(fp)
    with open('/home/signboard_retrieval/pkl/q_crop_list.pkl', 'rb') as fp: #len은 동일
        q_crop_list = pickle.load(fp)
    
    q_panorama_id = 0
    db_panorama_id = 0
    main(q_crop_list, db_crop_list, q_panorama_id, db_panorama_id, result_path=args.result_path, batch_size=args.batch_size,\
         num_workers=args.num_workers, device=args.device)
    
