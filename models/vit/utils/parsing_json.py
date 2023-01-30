import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw 
import natsort


def cropping_polygon(img_path, polygon, cnt):
    
    try:
        img = cv2.imread(img_path)
        pts = np.array(polygon)
        pts = pts.astype(int)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped_img = img[y:y+h, x:x+w].copy()
        cv2.imwrite('./01.cropped/'+img_path[5:-4] + '_' + str(cnt) + '.jpg',cropped_img)
    except:
        pass
    
    print(f'read_img_path done')


def parsing_json(json_path, img_path):
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        cnt=0
        for i,data_sh in enumerate(data['shapes']):
            if data['shapes'][i]['flags']['ganpan_object']==True:
                polygon = data['shapes'][i]['points']
                cropping_polygon(img_path, polygon, cnt)
                cnt+=1
        print(f'cnt == {cnt}')
        
    pass

if __name__ == "__main__":
    root_dir = "./"
    json_dir=os.path.join(root_dir, "annotation/")
    img_dir=os.path.join(root_dir, "img/")
    json_annotation_list = os.listdir(json_dir)
    img_list = os.listdir(img_dir)
    img_list = natsort.natsorted(img_list) #['간판_세로형간판_025611.JPG', '간판_세로형간판_025613.jpg', '간판_세로형간판_025614.JPG', '간판_세로형간판_025615.jpg']
    json_annotation_list = natsort.natsorted(json_annotation_list) #['간판_세로형간판_025611.json', '간판_세로형간판_025612.json', '간판_세로형간판_025613.json', '간판_세로형간판_025615.json']

    for json_file_name in json_annotation_list:
        json_path = os.path.join(json_dir, json_file_name)   
        img_file_name = json_file_name[:-5]+'.jpg' 
        if img_file_name in img_list:
            img_path=os.path.join(root_dir, "img/", img_file_name)
            print(f'lets get start about json_path and img_path and {json_path, img_path}')        
            res = parsing_json(json_path, img_path)
        else:
            pass
            
