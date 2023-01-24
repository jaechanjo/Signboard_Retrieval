import cv2
import json
import os
from pathlib import Path

def dic2visualization(q_img_path, db_img_path, q_json_path, db_json_path, result_dict,\
                      save_dir, method='vit'):
    
    panorama_id = Path(q_json_path).stem.split('@')[0] #str
    
    q_panorama = cv2.imread(q_img_path)
    
    db_panorama = cv2.imread(db_img_path)
    
    # horizontally concatenates images
    # of same height
    im_h = cv2.hconcat([q_panorama, db_panorama])
    
    # horizontal length
    h = q_panorama.shape[1]
    
    with open(q_json_path, "r") as js:
        q_lbl = json.load(js)
        
    with open(db_json_path, "r") as js:
        db_lbl = json.load(js)
    
    for q_idx in result_dict[panorama_id].keys():
        
        coord_list = q_lbl['shapes'][int(q_idx)]['points']
        x_list = list(map(lambda x: int(x[0]), coord_list))
        y_list = list(map(lambda x: int(x[1]), coord_list))
        del coord_list

        qmin_x = min(x_list)
        qmax_x = max(x_list)
        del x_list

        qmin_y = min(y_list)
        qmax_y = max(y_list)
        del y_list
        
        #draw rectangle
        if len(result_dict[panorama_id][q_idx]) == 0:#empty
            cv2.rectangle(im_h, (qmin_x,qmin_y), (qmax_x,qmax_y), (0,0,255), 2) #cv2 (B,G,R)
        else:
            cv2.rectangle(im_h, (qmin_x,qmin_y), (qmax_x,qmax_y), (0,255,0), 2)
        
            for db_idx in result_dict[panorama_id][q_idx]:

                coord_list = db_lbl['shapes'][int(db_idx)]['points']
                x_list = list(map(lambda x: int(x[0]), coord_list))
                y_list = list(map(lambda x: int(x[1]), coord_list))
                del coord_list

                dbmin_x = min(x_list)
                dbmax_x = max(x_list)
                del x_list

                dbmin_y = min(y_list)
                dbmax_y = max(y_list)
                del y_list
                
                #draw_rectangle
                cv2.rectangle(im_h, (dbmin_x+h,dbmin_y), (dbmax_x+h,dbmax_y), (0,255,0), 2)
                
                #draw line
                cv2.line(im_h, (qmax_x, int((qmin_y+qmax_y)/2)), (dbmin_x+h, int((dbmin_y+dbmax_y)/2)), (0,255,0), 2)
    
    #save
    out_file = save_dir + f"/pair_{panorama_id}_{method}.jpg"
    
    #if duplicated, remove it
    if os.path.isfile(out_file):
        os.remove(out_file)
    
    cv2.imwrite(save_dir + f"/pair_{panorama_id}_{method}.jpg", im_h)
    
    #done
    print(f"\nSaving...\n{out_file}\nDone.")