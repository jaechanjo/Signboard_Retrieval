import argparse
import os
import time
import faiss
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import natsort
import pickle5 as pickle
import numpy as np

from torch.utils.data import  DataLoader
from torchvision import models 
from torchinfo import summary
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm

from vit.dataset import AugmentedDataset
from vit.model import vit_base_patch8_224_dino, beitv2_large_patch16_224_in22k, swin_large_patch4_window12_384_in22k

@torch.no_grad()
def features_extract(model, img_list, batch_size, num_workers, device):
    features = []

    dataset = AugmentedDataset(img_list, img_size=384)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc='extracting', unit='batch')
    
    start = time.time()
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(device)
        feat = model(imgs).cpu()
        print(f'feat shape is {feat.shape}')
        features.append(feat)
    print(feat.shape[0])
    print(f'feature extraction: {time.time() - start:.2f} sec')

    start = time.time()
    feature = np.vstack(features)
    print(f"feature shape is {feature.shape}")

    feature = feature.reshape(feature.shape[0],-1)
    print(f"new_feature shape is {feature.shape}")
    
    print(f'convert to numpy: {time.time() - start:.2f} sec')

    start = time.time()
    feature = torch.from_numpy(feature)
    print(f'convert to tensor: {time.time() - start:.2f} sec')

    start = time.time()
    
    return feature

class FeatLayer:
    def __init__(self):
        self.model = model

    def build_model(model, q_crop_list, db_crop_list, batch_size, num_workers, device):
        # print(f"{summary(model, input_size=(1, 3, 384, 384))}")
        ############# device check #############
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        ############ 쿼리/ 레퍼런스 이미지 인덱스 폴더별 피처뽑기 #######
        #query feature
        query_feature = features_extract(model, q_crop_list, batch_size, num_workers, device='cuda')
        print(f"===== Done: query_feature\n\n")
        #db feature
        reference_feature = features_extract(model, db_crop_list, batch_size, num_workers, device='cuda')
        print(f"===== Done: db_feature\n\n")
        
        return query_feature, reference_feature

############# create ATTN model #############
def attn(q_crop_list, db_crop_list, batch_size, num_workers, device):    
   
    model = swin_large_patch4_window12_384_in22k()
    query_feature, reference_feature = FeatLayer.build_model(model, q_crop_list, db_crop_list, batch_size, num_workers, device)
    del model
    
    return  query_feature, reference_feature

def main(q_crop_list, db_crop_list, batch_size, num_workers, device):

    query_feature, reference_feature = attn(q_crop_list, db_crop_list, batch_size, num_workers, device)

    return query_feature, reference_feature
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')    
    args = parser.parse_args()