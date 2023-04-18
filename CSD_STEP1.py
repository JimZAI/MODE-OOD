#! /usr/bin/env python3
import torch
import os
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args.ckp)

M = 6
loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True)
batch_size = args.batch_size

from models.attention import AttentionSimilarity
checkpoint = torch.load(args.checkpoint)
attention = AttentionSimilarity(hidden_size=512, inner_size=args.feat_dim)
attention.load_state_dict(checkpoint['attention'])
attention.cuda()
attention.eval()

FORCE_RUN = True
for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:
    cache_name = f"cache/{args.in_dataset}_{split}_{args.name}_in_alllayers.pth"
    if FORCE_RUN or not os.path.exists(cache_name):
        feat_log = torch.zeros([int(len(in_loader.dataset)), 512, M])
        score_log = torch.zeros([len(in_loader.dataset), num_classes])
        label_log = torch.zeros([len(in_loader.dataset)])
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = int(batch_idx * batch_size)
            end_ind = int(min((batch_idx + 1) * batch_size, len(in_loader.dataset)))
            score, feature_list = model.feature_list(inputs)

            feature_list = feature_list[: int(batch_size)]
            feature_list_4 = torch.cat([F.adaptive_avg_pool2d(feature_list, 2).squeeze()], dim=1)
            feature_list_4 = torch.reshape(feature_list_4, (feature_list_4.shape[0], 512, feature_list_4.shape[2] * feature_list_4.shape[3]))
            # global
            feature_list_1 = torch.cat([F.adaptive_avg_pool2d(feature_list, 1).squeeze(-1)], dim=1)
            # center
            feature_list_2 = feature_list[:,:,1:3,1:3]
            feature_list_2 = torch.cat([F.adaptive_avg_pool2d(feature_list_2, 1).squeeze(-1)], dim=1)
            # cat
            feature_list = torch.cat([feature_list_4, feature_list_1], dim=-1)
            feature_list = torch.cat([feature_list, feature_list_2], dim=-1)
            feat_log[start_ind:end_ind, :] = feature_list.detach()
            
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(in_loader)}")
        torch.save(feat_log,cache_name)
    else:
        feat_log=torch.load(cache_name)

for ood_dataset in args.out_datasets:
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.pth"
    if FORCE_RUN or not os.path.exists(cache_name):
        if args.attention:
            if args.spatial:
                ood_feat_log = torch.zeros([len(out_loader.dataset), 512, M])
            else:
                ood_feat_log = torch.zeros([len(in_loader.dataset), 80])
        else:
            ood_feat_log = torch.zeros([len(out_loader.dataset), 512])
        ood_score_log = torch.zeros([len(out_loader.dataset), num_classes])
        model.eval()
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))
            score, feature_list = model.feature_list(inputs)
            feature_list_4 = torch.cat([F.adaptive_avg_pool2d(feature_list, 2).squeeze()], dim=1)
            feature_list_4 = torch.reshape(feature_list_4, (feature_list_4.shape[0], 512, feature_list_4.shape[2] * feature_list_4.shape[3]))
            # global
            feature_list_1 = torch.cat([F.adaptive_avg_pool2d(feature_list, 1).squeeze(-1)], dim=1)
            # center
            feature_list_2 = feature_list[:,:,1:3,1:3]
            feature_list_2 = torch.cat([F.adaptive_avg_pool2d(feature_list_2, 1).squeeze(-1)], dim=1)
            # cat
            feature_list = torch.cat([feature_list_4, feature_list_1], dim=-1)
            feature_list = torch.cat([feature_list, feature_list_2], dim=-1)
            ood_feat_log[start_ind:end_ind, :] = feature_list.detach()
            ood_score_log[start_ind:end_ind] = score.detach()
            
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(out_loader)}")
        torch.save(ood_feat_log, cache_name)
    else:
        ood_feat_log=torch.load(cache_name)
print(time.time() - begin)