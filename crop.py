import cv2
import json
import numpy as np
from pathlib import Path

def crop_get(q_img_path, db_img_path, q_json_path, db_json_path):
    
    panorama_id = Path(q_json_path).stem.split('@')[0] #str
    
    q_panorama = cv2.cvtColor(cv2.imread(q_img_path), cv2.COLOR_BGR2RGB)
    
    #get query cropped image
    q_crop_list = [] #많아봤자 10개 이미지, 작은 크기이므로
    with open(q_json_path, "r") as js:
        lbl = json.load(js)

        #read label
        for i in range(len(lbl['shapes'])):

            # crop image
            coord_list = lbl['shapes'][i]['points']
            x_list = list(map(lambda x: int(x[0]), coord_list))
            y_list = list(map(lambda x: int(x[1]), coord_list))
            del coord_list

            min_x = min(x_list)
            max_x = max(x_list)
            del x_list

            min_y = min(y_list)
            max_y = max(y_list)
            del y_list

            croppedImage = q_panorama[min_y: max_y, min_x: max_x]

            #get cropped Image
            q_crop_list.append(croppedImage)
            del croppedImage
        
    db_panorama = cv2.cvtColor(cv2.imread(db_img_path), cv2.COLOR_BGR2RGB)
        
    #get db cropped image
    db_crop_list = [] #많아봤자 10개 이미지, 작은 크기이므로
    with open(db_json_path, "r") as js:
        lbl = json.load(js)

        #read label
        for i in range(len(lbl['shapes'])):

            # crop image
            coord_list = lbl['shapes'][i]['points']
            x_list = list(map(lambda x: int(x[0]), coord_list))
            y_list = list(map(lambda x: int(x[1]), coord_list))
            del coord_list

            min_x = min(x_list)
            max_x = max(x_list)
            del x_list

            min_y = min(y_list)
            max_y = max(y_list)
            del y_list
            
            croppedImage = db_panorama[min_y: max_y, min_x: max_x]
            
            #get cropped Image
            db_crop_list.append(croppedImage)
            del croppedImage
        
    return q_crop_list, db_crop_list, panorama_id