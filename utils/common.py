import numpy as np
import cv2
import kornia as K
import torch

def mask_out(bboxes, image):
    np_mask = np.zeros_like(image)
    for item in bboxes:
        min_x, min_y = item['points'][0]
        max_x, max_y = item['points'][2]
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        db_mask = cv2.rectangle(np_mask, (min_x, min_y), (max_x, max_y), 255, -1)
    image = cv2.bitwise_and(image, db_mask)
    return image


def draw_box(out, boxes, color, margin, W0, H0, right=False, matched=False):
    for box in boxes:
        (min_x, min_y), (max_x, max_y) = box['points'][0], box['points'][2]
        box_w, box_h = max_x - min_x, max_y - min_y
        if right:
            min_x = min_x + margin + W0
        cv2.rectangle(out, (int(min_x), int(min_y), int(box_w), int(box_h)), color, 3)


def visualize(image0, image1, boxes0, boxes1, matches):
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    margin = 10
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1
    red, green = (0, 0, 255), (0, 255, 0)
    draw_box(out, boxes0, red, margin, W0, H0)
    draw_box(out, boxes1, red, margin, W0, H0, right=True)

    for match in matches:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        db_box, query_box = boxes0[match['db_box_index']], boxes1[match['query_box_index']]
        (min_x0, min_y0), (max_x0, max_y0) = db_box['points'][0], db_box['points'][2]
        box_w0, box_h0 = max_x0 - min_x0, max_y0 - min_y0
        x_center0, y_center0 = int((min_x0 + max_x0) / 2), int((min_y0 + max_y0) / 2)
        (min_x1, min_y1), (max_x1, max_y1) = query_box['points'][0], query_box['points'][2]
        box_w1, box_h1 = max_x1 - min_x1, max_y1 - min_y1
        x_center1, y_center1 = int((min_x1 + max_x1) / 2), int((min_y1 + max_y1) / 2)
        cv2.rectangle(out, (int(min_x0), int(min_y0), int(box_w0), int(box_h0)), green, 3)
        cv2.rectangle(out, (int(min_x1 + margin + W0), int(min_y1), int(box_w1), int(box_h1)), green, 3)
        cv2.line(out, (int(x_center0), int(y_center0)), (int(x_center1 + margin + W0), int(y_center1)), color,
                 thickness=2, lineType=cv2.LINE_AA)
    return out


def idx_pts_bbox(pts, xyxy):
    ret = []
    min_x, min_y, max_x, max_y = xyxy
    for i in range(len(pts)):
        x, y = pts[i]
        if x >= min_x and x <= max_x and y >= min_y and y <= max_y:
            ret.append(i)
    return ret


def load_torch_image(image):
    img = K.image_to_tensor(image, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


# colored image
def read_image(img_path):
    ret = cv2.imread(str(img_path))
    ret = cv2.resize(ret, (2000, 1000))
    return ret


# gray scaled image
def read_gray_image(img_path):
    ret = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    ret = cv2.resize(ret, (2000, 1000))
    return ret