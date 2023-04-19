import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
import tables
from numpy import *
import torch.nn.functional as F
torch.manual_seed(1)
torch.cuda.manual_seed(1)
pdist = torch.nn.PairwiseDistance(p=2)
np.random.seed(1)
from tqdm import tqdm
import time

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
args = get_args()
Global = True

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.pth"
feat_log=torch.load(cache_name)
feat_log = feat_log.permute(0, 2, 1) 
feat_log = feat_log.permute(1, 0, 2)  
feat_log = torch.reshape(feat_log, (feat_log.shape[0] * feat_log.shape[1], 512)) 

cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.pth"
feat_log_val= torch.load(cache_name)
feat_log_val = feat_log_val.permute(0, 2, 1) 
feat_log_val = feat_log_val.permute(1, 0, 2)  

ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.pth"
    ood_feat_log=torch.load(cache_name)
    ood_feat_log = ood_feat_log.permute(0, 2, 1)  
    ood_feat_log = ood_feat_log.permute(1, 0, 2) 
    ood_feat_log_all[ood_dataset] = ood_feat_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x)) 

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


#################### KNN score OOD detection #################
index = faiss.IndexFlatL2(ftrain.shape[1])
index.add(ftrain)
for K in [10]:
    print(" ### ID_train vs ID_val")
    DD = []
    for s in tqdm(range(ftest.shape[0])):
        D, _ = index.search(ftest[s], K)
        scores = -D[:,-1]
        scores = torch.from_numpy(scores)
        DD.append(scores)
    DD_tensor = torch.stack(DD) # 5 10000
    DD_tensor_max, _ = DD_tensor.max(0) # 10000
    scores_in = DD_tensor_max.numpy()

    all_results = []
    all_score_ood = []

    for ood_dataset, food in food_all.items():
        t0 = time.time()
        print(f"### ID-train vs {ood_dataset} ###")
        DD = []
        for s in tqdm(range(food.shape[0])):
            D, _ = index.search(food[s], K)
            scores = -D[:, -1]
            scores = torch.from_numpy(scores)
            DD.append(scores)
        DD_tensor = torch.stack(DD)
        DD_tensor_max, _ = DD_tensor.max(0)
        scores_ood_test = DD_tensor_max.numpy()

        all_score_ood.extend(scores_ood_test)
        results = metrics.cal_metric(scores_in, scores_ood_test)
        print(time.time() - t0)
        all_results.append(results)

    metrics.print_all_results(all_results, args.out_datasets, f'KNN k={K}')
    print()

