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
# import dircache
import time

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, idx_pts_bbox, frame2tensor)
from utils.common import mask_out, read_gray_image, read_image, visualize
torch.set_grad_enabled(False)


def main(db_path, query_path, max_keypoints, keypoint_threshold, superglue, nms_radius,
         sinkhorn_iterations, match_threshold, k, bool_visualize, output_dir):
    db_path, query_path = Path(db_path), Path(query_path)
    result = {'db_image': str(db_path), 'query_image': str(query_path)}

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    db_boxes = []
    db_det_fname = os.path.splitext(db_path)[0] + '.json'
    query_boxes = []
    query_det_fname = os.path.splitext(query_path)[0] + '.json'
    with open(db_det_fname, 'r') as f:
        json_data = json.load(f)
        db_boxes = json_data['shapes']
    with open(query_det_fname, 'r') as f:
        json_data = json.load(f)
        query_boxes = json_data['shapes']

    org_img0, org_img1 = read_image(db_path), read_image(query_path)
    db_image, query_image = read_gray_image(db_path), read_gray_image(query_path)

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

    match_list = []
    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    for q_idx, query_box in enumerate(query_boxes):
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
            result_item = {'db_box_index': db_box_matched, 'query_box_index': q_idx}
            match_list.append(result_item)

        result['matches'] = match_list
    if bool_visualize:
        out = visualize(org_img0, org_img1, db_boxes, query_boxes, result['matches'])
        if output_dir is not None:
            Path(output_dir).mkdir(exist_ok=True)
            cv2.imwrite(output_dir + '/result.jpg', out)
    return result


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
    parser.add_argument(
        '--visualize', action='store_true',
        help='Visualize matched boxes'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to put visualized output'
    )
    opt = parser.parse_args()
    print(opt)

    start = time.time()
    result = main(opt.db_path, opt.query_path, opt.max_keypoints, opt.keypoint_threshold,
         opt.superglue, opt.nms_radius, opt.sinkhorn_iterations, opt.match_threshold,
         opt.k, opt.visualize, opt.output_dir)
    end = time.time()
    print(f"inference time : {end - start : .3f} sec")
    print(result)
