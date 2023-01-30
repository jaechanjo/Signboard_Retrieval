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

from models.vit.dataset import AugmentedDataset_val 
from models.vit.model import vit_base_patch8_224_dino, beitv2_large_patch16_224_in22k, swin_large_patch4_window12_384_in22k

#evaluation
@torch.no_grad()
def features_extract_val(model, img_path_list, batch_size, num_workers, device):
    features = []

    dataset = AugmentedDataset_val(img_path_list, img_size=384)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    model.eval()
    bar = tqdm(loader, ncols=120, desc='extracting', unit='batch')
    
    for batch_idx, batch_item in enumerate(bar):
        imgs = batch_item['img'].to(device)
        feat = model(imgs).cpu()
        features.append(feat)

    feature = np.vstack(features)
    feature = feature.reshape(feature.shape[0],-1)
    feature = torch.from_numpy(feature)
    
    return feature

class FeatLayer_val:
    def __init__(self):
        self.model = model

    def build_model(model, q_path_list, db_path_list, batch_size, num_workers, device):
        
        ############# device check #############
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        ############ extract featuer ###########
        #query feature
        query_feature = features_extract_val(model, q_path_list, batch_size, num_workers, device='cuda')
        print("\nExtracting query feature...\nDone.")
        #db feature
        reference_feature = features_extract_val(model, db_path_list, batch_size, num_workers, device='cuda')
        print("\nExtracting db featrue...\nDone.")
        
        return query_feature, reference_feature
    

############# create ATTN model #############
def attn(q_path_list, db_path_list, batch_size, num_workers, device):    
   
    model = swin_large_patch4_window12_384_in22k()
    query_feature, reference_feature = FeatLayer_val.build_model(model, q_path_list, db_path_list, batch_size, num_workers, device)
    del model
    
    return  query_feature, reference_feature

def main(q_path_list, db_path_list, batch_size, num_workers, device):

    query_feature, reference_feature = attn(q_path_list, db_path_list, batch_size, num_workers, device)

    return query_feature, reference_feature
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--result_path', type=str, default='/home/signboard_retrieval/result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')    
    args = parser.parse_args()