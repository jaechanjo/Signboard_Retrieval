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
import time
# import dircache

from utils.common import mask_out, frame2tensor, read_gray_image, load_torch_image, idx_pts_bbox, visualize, read_image
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
torch.set_grad_enabled(False)


def main(db_path, query_path, resize_factor, match_threshold, k, bool_visualize, output_dir):
    config['match_coarse']['thr'] = match_threshold
    db_path, query_path = Path(db_path), Path(query_path)
    result = {'db_image': str(db_path), 'query_image': str(query_path)}

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    matching = KF.LoFTR(pretrained='outdoor', config=config).eval().to(device)

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
    db_image = read_image(db_path)
    query_image = read_image(query_path)

    # masking out
    db_image = mask_out(db_boxes, db_image)
    query_image = mask_out(query_boxes, query_image)

    H, W = db_image.shape[0], db_image.shape[1]
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
    confidences = correspondences['confidence'].cpu().numpy()
    color = cm.jet(confidences)
    match_list = []

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
        '--resize_float', default=0.8,
        help='Resize the image')
    parser.add_argument(
        '--match_threshold', type=float, default=0.4,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--k', type=int, default=2,
        help='Box matching threshold points'
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
    result = main(opt.db_path, opt.query_path, opt.resize_float, opt.match_threshold,
                  opt.k, opt.visualize, opt.output_dir)
    end = time.time()
    print(f"inference time : {end - start : .3f} sec")
    # print(result)
