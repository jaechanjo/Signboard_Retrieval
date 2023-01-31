from pathlib import Path
from PIL import Image
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
import json
import time
# import dircache

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, idx_pts_bbox, frame2tensor)
from utils.common import mask_out, read_gray_image
torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--db_path', type=str, default='data/sample/gt/db', help='Path to db dir'
    )
    parser.add_argument(
        '--query_path', type=str, default='data/sample/gt/query', help='Path to query dir'
    )
    parser.add_argument(
        '--max_keypoints', type=int, default=1000,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
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

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

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

        db_image = read_gray_image(db_path / name0)
        query_image = read_gray_image(query_path / name1)

        # masking out
        db_image = mask_out(db_boxes, db_image)
        query_image = mask_out(query_boxes, query_image)

        inp0 = frame2tensor(db_image, device)
        inp1 = frame2tensor(query_image, device)

        do_match = True
        if do_match:
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

        # count signboards in db
        cnt_dbsign += len(db_boxes)
        cnt_qsign += len(query_boxes)

        for item in db_boxes:
            if item['flags']['matched'] == True:
                cnt_matched += 1

        result_item = []
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
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
    print(f"F1-score : {2 * (recall * precision) / (recall + precision)}")
    # db_unmatched = cnt_dbsign - cnt_matched
    # q_unmatched = cnt_qsign - cnt_matched
    # print('num signs in db : ', cnt_dbsign)
    # print('num signs in query : ', cnt_qsign)
    # print('num matched : ', cnt_matched)
    # print('num unmatched in db : ', db_unmatched)
    # print('num unmatched in query : ', q_unmatched)
    print(f'eval time : {end - start : .3f} sec')
