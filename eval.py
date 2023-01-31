import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import random
import torch
import os
import json
# import dircache
import time

from utils.common import mask_out, frame2tensor, read_gray_image, load_torch_image, idx_pts_bbox, read_image
torch.set_grad_enabled(False)
config = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        'attention': 'linear',
        'temp_bug_fix': False,
    },
    'match_coarse': {
        'thr': 0.7,
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': True,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
    },
    'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear'},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db_path', type=str, default='data/gt/db', help='Path to db dir'
    )
    parser.add_argument(
        '--query_path', type=str, default='data/gt/query', help='Path to query dir'
    )
    parser.add_argument(
        '--resize_float', default=0.8,
        help='Resize the image')
    parser.add_argument(
        '--match_threshold', type=float, default=0.4,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--k', type=int, default=2,
        help='Box matching threshold'
    )

    opt = parser.parse_args()
    print(opt)
    start = time.time()
    config['match_coarse']['thr'] = opt.match_threshold
    db_image_list = []
    query_image_list = []
    for item in os.listdir(opt.db_path):
        file_name, file_extention = os.path.splitext(item)
        if file_extention in '.jpg':
            db_image_list.append(item)
    for item in os.listdir(opt.query_path):
        file_name, file_extention = os.path.splitext(item)
        if file_extention in '.jpg':
            query_image_list.append(item)

    image_pairs = []
    result = {}
    for db_item in db_image_list:
        match_num = db_item.split('@')[0]
        for q_item in query_image_list:
            if q_item.split('@')[0] == match_num:
                image_pairs.append([db_item, q_item])

    # Load the LoFTR model.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    matching = KF.LoFTR(pretrained='outdoor', config=config).eval().to(device)

    db_path = Path(opt.db_path)
    query_path = Path(opt.query_path)
    k = opt.k
    num_FP, num_TP, num_GT = 0, 0, 0
    db_unmatched, q_unmatched, cnt_matched, cnt_dbsign, cnt_qsign = 0, 0, 0, 0, 0
    # match to unmatch, unmatch to match, unmatch to unmatch
    m2u, u2m, u2u = 0, 0, 0
    TP_list = []
    FP_list = []
    GT_list = []
    for i, pair in enumerate(image_pairs):
        name0, name1 = pair[:2]
        pair_idx = name0.split('@')[0]
        db_boxes = []
        db_gt_fname = name0.split('.')[0] + '.json'
        query_boxes = []
        query_gt_fname = name1.split('.')[0] + '.json'
        with open(db_path / db_gt_fname, 'r') as f:
            json_data = json.load(f)
            db_boxes = json_data['shapes']
        with open(query_path / query_gt_fname, 'r') as f:
            json_data = json.load(f)
            query_boxes = json_data['shapes']

        db_image = read_image(db_path / name0)
        query_image = read_image(query_path / name1)

        # masking out
        db_image = mask_out(db_boxes, db_image)
        query_image = mask_out(query_boxes, query_image)

        H, W = db_image.shape[0], db_image.shape[1]
        resize_factor = opt.resize_float
        db_image = cv2.resize(db_image, (int(W * resize_factor), int(H * resize_factor)))
        query_image = cv2.resize(query_image, (int(W * resize_factor), int(H * resize_factor)))

        inp0 = load_torch_image(db_image)
        inp1 = load_torch_image(query_image)

        input_dict = {"image0": K.color.rgb_to_grayscale(inp0).to(device),  # LofTR works on grayscale images only
                      "image1": K.color.rgb_to_grayscale(inp1).to(device)}

        with torch.inference_mode():
            correspondences = matching(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy() / resize_factor
        mkpts1 = correspondences['keypoints1'].cpu().numpy() / resize_factor

        # count signboards in db
        cnt_dbsign += len(db_boxes)
        cnt_qsign += len(query_boxes)

        for item in db_boxes:
            if item['flags']['matched'] == True:
                cnt_matched += 1

        result_item = []

        # print('num points : ', len(mkpts0))
        TP_list.append([name0, name1])
        FP_list.append([name0, name1])
        for q_idx, query_box in enumerate(query_boxes):
            if query_box['label'] != "100": num_GT += 1
            query_box_pts = query_box['points']
            query_xyxy = (query_box_pts[0][0], query_box_pts[0][1],
                          query_box_pts[2][0], query_box_pts[2][1])
            pts_in_box = idx_pts_bbox(mkpts1, query_xyxy)
            npts = len(pts_in_box)
            db_box_matched = -1
            max_match = 0
            for d_idx, db_box in enumerate(db_boxes):
                db_box_pts = db_box['points']
                db_xyxy = (db_box_pts[0][0], db_box_pts[0][1],
                           db_box_pts[2][0], db_box_pts[2][1])
                num_matched = len(idx_pts_bbox(mkpts0[pts_in_box], db_xyxy))
                if max_match < num_matched:
                    max_match = num_matched
                    db_box_matched = d_idx
            if db_box_matched != -1 and max_match >= k:
                db_pred = db_boxes[db_box_matched]["label"]
                q_pred = query_boxes[q_idx]["label"]
                result_item.append([db_box_matched, q_idx])
                if db_pred == q_pred and (db_pred != "100" and q_pred != "100"):
                    num_TP += 1
                    TP_list.append([db_pred, q_pred])
                else:
                    num_FP += 1
                    FP_list.append([db_pred, q_pred])
                    if q_pred == "100" and db_pred != "100":
                        u2m += 1
                    elif db_pred == "100" and q_pred != "100":
                        m2u += 1
                    elif db_pred == "100" and q_pred == "100":
                        u2u += 1
        result[pair_idx] = result_item
    end = time.time()
    precision = num_TP / (num_TP + num_FP)
    recall = num_TP / num_GT
    print('num GT : ', num_GT)
    print('num TP, num FP', num_TP, num_FP)
    print(f"recall : {recall : .3f}")
    print(f"precision : {precision : .3f}")
    print(f"F1-score : {2*(recall * precision) / (recall + precision)}")
    # db_unmatched = cnt_dbsign - cnt_matched
    # q_unmatched = cnt_qsign - cnt_matched
    # print('num signs in db : ', cnt_dbsign)
    # print('num signs in query : ', cnt_qsign)
    # print('num matched : ', cnt_matched)
    # print('num unmatched in db : ', db_unmatched)
    # print('num unmatched in query : ', q_unmatched)
    print(f'eval time : {end - start : .3f} sec')
