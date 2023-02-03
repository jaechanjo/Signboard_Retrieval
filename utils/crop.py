import cv2
import json
import numpy as np
from pathlib import Path


def crop_get(q_img_path, db_img_path, q_json, db_json):
    
    '''
    q_img_path : str, query panorama image path
    db_img_path : str, db panorama image path
    q_json : json, loaded query json file
    db_json : json, loaded db json file
    '''
    
    q_panorama_id = Path(q_img_path).stem.split('@')[0] #str
    db_panorama_id = Path(db_img_path).stem.split('@')[0] #str
    
    q_panorama = cv2.cvtColor(cv2.imread(q_img_path), cv2.COLOR_BGR2RGB)
    
    #get query cropped image
    q_crop_list = [] #많아봤자 10개 이미지, 작은 크기이므로

    #read label
    for i in range(len(q_json['shapes'])):

        # crop image
        coord_list = q_json['shapes'][i]['points']
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

    #read label
    for i in range(len(db_json['shapes'])):

        # crop image
        coord_list = db_json['shapes'][i]['points']
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
        
    return q_crop_list, db_crop_list, q_panorama_id, db_panorama_id