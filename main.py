import matplotlib.cm as cm
import json
import os
from pathlib import Path

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.cm as cm
import torch

# import dircache
from cfg.config.params import *
from utils.common import mask_out, load_torch_image, idx_pts_bbox, read_image

torch.set_grad_enabled(False)


class LoFTR:
    def __init__(self):
        self.config = LoFTR_config
        self.k = k
        self.resize_float = resize_float

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.matching = KF.LoFTR(pretrained='outdoor', config=self.config).eval().to(self.device)

    def inference(self, db_path, query_path, db_det, query_det):
        db_path, query_path = Path(db_path), Path(query_path)
        result = {'db_image': str(db_path), 'query_image': str(query_path)}

        # db_boxes = []
        # db_det_fname = os.path.splitext(db_path)[0] + '.json'
        # query_boxes = []
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
        db_image = read_image(db_path)
        query_image = read_image(query_path)

        # masking out
        db_image = mask_out(db_boxes, db_image)
        query_image = mask_out(query_boxes, query_image)

        H, W = db_image.shape[0], db_image.shape[1]
        db_image = cv2.resize(db_image, (int(W * self.resize_float), int(H * self.resize_float)))
        query_image = cv2.resize(query_image, (int(W * self.resize_float), int(H * self.resize_float)))

        inp0 = load_torch_image(db_image)
        inp1 = load_torch_image(query_image)

        input_dict = {"image0": K.color.rgb_to_grayscale(inp0).to(self.device),  # LofTR works on grayscale images only
                      "image1": K.color.rgb_to_grayscale(inp1).to(self.device)}

        with torch.inference_mode():
            correspondences = self.matching(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy() / self.resize_float
        mkpts1 = correspondences['keypoints1'].cpu().numpy() / self.resize_float
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
            if db_box_matched != -1 and max_match >= self.k:
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

    # load model
    model = LoFTR()

    # inference
    result = model.inference(db_path, query_path, db_det, query_det)
    print(result)
