import cv2
import glob
from pathlib import Path
import json
import os

def parsing(query_dir, db_dir):
    
    for img_lbl_dir in [query_dir, db_dir]: #parsing both query and db
    
        for i, (img_path, lbl_path) in enumerate(zip(sorted(glob.glob(img_lbl_dir + '*.jpg')), sorted(glob.glob(img_lbl_dir + '*.json')))):

            #check match img with label
            img_id = int(Path(img_path).stem.split('@')[0])
            lbl_id = int(Path(lbl_path).stem.split('@')[0])
            if img_id == lbl_id:

                #read image origin
                img_org = cv2.imread(img_path)

                #open label
                with open(lbl_path, "r") as js:
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

                        croppedImage = img_org[min_y: max_y, min_x: max_x]

                        # match info
                        match = lbl['shapes'][i]['label'] #if matched specific id, unmatched 100
                        change = str(lbl['shapes'][i]['flags']['changed']) # False F, True T

                        #save cropped Image
                        save_dir = str(Path(img_lbl_dir).parent) + '/' + str(Path(img_lbl_dir).stem) + '_val/'
                        dir_cropped = save_dir + f"/{img_id}/" #####
                        if not os.path.exists(dir_cropped):
                            os.makedirs(dir_cropped, exist_ok=True)
                        cropped_fname = f"{str(i)}_{match}_{change}"
                        cv2.imwrite(dir_cropped + cropped_fname + '.jpg', croppedImage)
            else:
                raise Exception("ID Matching Error")
    print("\nParsing...\nDone.")
    
    #directory of parsing images
    query_val_dir = str(Path(query_dir).parent) + '/' + str(Path(query_dir).stem) + '_val/'
    db_val_dir = str(Path(db_dir).parent) + '/' + str(Path(db_dir).stem) + '_val/'
    
    return query_val_dir, db_val_dir