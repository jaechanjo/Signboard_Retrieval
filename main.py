import json
import os
from pathlib import Path

import torch

from cfg.config.params import *
from models.matching import Matching
from models.utils import (idx_pts_bbox, frame2tensor)
from utils.common import mask_out, read_gray_image, read_image

# import dircache

torch.set_grad_enabled(False)


class SuperGlue:

    def __init__(self):
        self.config = superglue_config
        self.k = k

        # device option
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.matching = Matching(self.config).eval().to(self.device)

    def inference(self, db_path, query_path, db_det, query_det):
        db_path, query_path = Path(db_path), Path(query_path)
        result = {'db_image': str(db_path), 'query_image': str(query_path)}

        # db_det_fname = os.path.splitext(db_path)[0] + '.json'
        # query_det_fname = os.path.splitext(query_path)[0] + '.json'
        # with open(db_det_fname, 'r') as f:
        #     json_data = json.load(f)
        #     db_boxes = json_data['shapes']
        # with open(query_det_fname, 'r') as f:
        #     json_data = json.load(f)
        #     query_boxes = json_data['shapes']
        db_boxes = db_det['shapes']
        query_boxes = query_det['shapes']

        org_img0, org_img1 = read_image(db_path), read_image(query_path)
        db_image, query_image = read_gray_image(db_path), read_gray_image(query_path)

        # masking out
        db_image = mask_out(db_boxes, db_image)
        query_image = mask_out(query_boxes, query_image)

        inp0 = frame2tensor(db_image, self.device)
        inp1 = frame2tensor(query_image, self.device)

        do_match = True
        if do_match:
            pred = self.matching({'image0': inp0, 'image1': inp1})
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
            if db_box_matched != -1 and max_match >= self.k:
                # db_pred = db_boxes[db_box_matched]["label"]
                # q_pred = query_boxes[q_idx]["label"]
                result_item = {'db_box_index': db_box_matched, 'query_box_index': q_idx}
                match_list.append(result_item)

            result['matches'] = match_list
        # if bool_visualize:
        #     out = visualize(org_img0, org_img1, db_boxes, query_boxes, result['matches'])
        #     if output_dir is not None:
        #         Path(output_dir).mkdir(exist_ok=True)
        #         cv2.imwrite(output_dir + '/result.jpg', out)
        return result


if __name__ == '__main__':
    db_path = 'data/sample/1/19.jpg'
    query_path = 'data/sample/1/21.jpg'
    db_json = 'data/sample/1/19.json'
    query_json = 'data/sample/1/21.json'
    with open(db_json, 'r') as f:
        db_det = json.load(f)
    with open(query_json, 'r') as f:
        query_det = json.load(f)

    model = SuperGlue()
    result = model.inference(db_path, query_path, db_det, query_det)
