import os.path as osp
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model_gt import CLIP, tokenize
from data import DataHelper, MolGraphDataset
from sklearn import preprocessing
import json
import os
from tqdm import tqdm
from utils import Logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


def cl_loss(s_features, t_features, args):
    if args.normalize:
        X = F.normalize(s_features, dim=-1)
        Y = F.normalize(t_features, dim=-1)

    if args.SSL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def main(args):
    setup_seed(seed)
    save_dir = "./res/{}/".format(args.data_name)
    logger = Logger(args, save_dir)
    model_save_name = f"{args.gnn_type}-{args.exp_time}-alignment.pkl"

    model = CLIP(args).to(device)
    # dataset = DataHelper(arr_edge_index, args)
    # in_g = Data(x=node_f, edge_index=edge_index).to(device)
    model.train()
    dataset = MolGraphDataset(args.graph_root)

    for j in range(args.epoch_num):
        epoch_loss = 0.0
        loader = pyg_DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            text = sample_batched[0]
            molecule_data = sample_batched[1].to(device)

            # s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
            # s_n_arr = s_n.numpy()  # .reshape((1, -1))
            # t_n_arr = t_n.numpy().reshape(-1)
            # s_n_text, t_n_text = [new_dict[i] for i in s_n_arr], [new_dict[j] for j in t_n_arr]
            # s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(
            #     t_n_text, context_length=args.context_length
            # ).to(device)

            # s_n, t_n = s_n.long().to(device), t_n.long().to(device)
            image_features, text_features = model(
                molecule_data, text, device
            )
            loss_01, acc_01 = cl_loss(text_features, image_features, args)
            loss_02, acc_02 = cl_loss(image_features, text_features, args)
            all_loss = (loss_01 + loss_02) / 2
            all_acc = (acc_01 + acc_02) / 2
            # node_loss = cal_cl_loss(s_image_features, s_text_features, labels)
            # gt_loss = cal_cl_loss(s_image_features, t_text_features, labels)
            # tt_loss = cal_cl_loss(s_text_features, t_text_features, labels)

            # all_loss = node_loss + args.edge_coef * gt_loss + args.edge_coef * tt_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)

            if i_batch % 100 == 0:
                logger.log("{}th loss in {} epoch:{}".format(i_batch, j + 1, loss))
            epoch_loss += loss / len(loader)
        # break
        logger.log("{}th epoch mean loss:{}".format(j + 1, epoch_loss))
    torch.save(model.state_dict(), osp.join(save_dir, model_save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epoch_num", type=int, default=2, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    parser.add_argument("--data_name", type=str, default="Cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log", type=int, default=1)
    # parser.add_argument("--graph_data_path", type=str, default="/data2/zhangyu/molecule-data/func_group_data/woshi_test_mols_graph_data.npy")
    parser.add_argument("--graph_root", type=str, default="")
    # gt config
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gt_layers", type=int, default=3)
    parser.add_argument("--att_d_model", type=int, default=128)
    parser.add_argument("--att_norm", type=bool, default=True)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--if_pos", type=bool, default=False)

    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G"])
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    # parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument('--text_pretrain_folder', type=str, default='./data/pretrained_SciBERT')
    parser.add_argument('--graph_pretrain_folder', type=str, default='./data/pretrained_GraphMVP')

    ########## for contrastive SSL ##########
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    ########## for SciBERT ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)

    start = time.perf_counter()
    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
