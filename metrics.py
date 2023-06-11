import rsatoolbox

import argparse
import pickle
import json
from tqdm import tqdm
import os, subprocess
import h5py
import random
import collections
import time

import numpy as np

from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

def create_target_rdm(data_dir):
    # read prob3
    h5_file = h5py.File(os.path.join(data_dir, "prob3.h5py"), "r")
    y_prob3_fg = h5_file.get('fg_classes')[()]
    y_prob3_bg = h5_file.get('bg_classes')[()]
    y_prob3 = np.zeros(y_prob3_fg.shape)
    y_prob3[:1000] = y_prob3_fg[:1000]
    y_prob3[1000:] = y_prob3_bg[1000:]+10
    h5_file.close()

    # obj-scene-modular
    half_length = len(y_prob3) // 2
    ref_matrix = np.ones((half_length*2, half_length*2))
    ref_matrix[:half_length,:half_length] = np.zeros((half_length, half_length))
    ref_matrix[half_length:,half_length:] = np.zeros((half_length, half_length))
    # plt.imshow(ref_matrix, cmap='Greys')
    # plt.axis("off")
    # plt.title("reference matrix: obj-scene-modular")
    # plt.show()

    # category-specific
    tmp = []
    for i in range(len(y_prob3)):
        for j in range(len(y_prob3)):
            tmp.append(y_prob3[i] - y_prob3[j])
    tmp = (np.array(tmp)!=0).astype(int).reshape((len(y_prob3), len(y_prob3)))
    ref_matrix_cat_specific = tmp
    # plt.imshow(ref_matrix_cat_specific, cmap='Greys')
    # plt.axis("off")
    # plt.title("reference matrix: category-specific")
    # plt.show()

    # co-occurrence
    tmp = []
    for i in range(len(y_prob3)):
        for j in range(len(y_prob3)):
            if y_prob3[i] > y_prob3[j]:
                a = y_prob3[i]
                b = y_prob3[j]
            else:
                b = y_prob3[i]
                a = y_prob3[j]
            if a-b == 10:
                tmp.append(0)
            elif a == b:
                tmp.append(0)
            else:
                tmp.append(1)
    tmp = np.array(tmp).astype(int).reshape((len(y_prob3), len(y_prob3)))
    ref_matrix_co_occur = tmp
    # plt.imshow(ref_matrix_co_occur, cmap='Greys')
    # plt.axis("off")
    # plt.title("reference matrix: co-occurrence")
    # plt.show()
    return [ref_matrix, ref_matrix_cat_specific, ref_matrix_co_occur]

def compute_RSA_metrics(args, saved_dict, target_rdms, FEATS, prob_name):
    ref_matrix, ref_matrix_cat_specific, ref_matrix_co_occur = target_rdms
    # record modularity metrics
    scaler = StandardScaler()
    for layer_name, feats in FEATS.items():
        X = feats 
        num_samples = len(X)

        # RSA 
        print("==> running RSA...")
        data = rsatoolbox.data.Dataset(X.reshape(num_samples,-1))
        rdms = rsatoolbox.rdm.calc_rdm(data)
        rdm_matrices = rdms.get_matrices().squeeze()
        _corr, _ = pearsonr(ref_matrix_cat_specific.reshape(-1), rdm_matrices.reshape(-1))
        saved_dict[args.exp_name][f"{prob_name}_rsa_cat_sep_{layer_name}"] = _corr
        _corr, _ = pearsonr(ref_matrix.reshape(-1), rdm_matrices.reshape(-1))
        saved_dict[args.exp_name][f"{prob_name}_rsa_obj_scene_{layer_name}"] = _corr
        _corr, _ = pearsonr(ref_matrix_co_occur.reshape(-1), rdm_matrices.reshape(-1))
        saved_dict[args.exp_name][f"{prob_name}_rsa_co_occure_{layer_name}"] = _corr
        
    return saved_dict

def compute_probing_metrics(args, saved_dict, FEATS, Y, prob_name):
    # record modularity metrics
    scaler = StandardScaler()
    for layer_name, feats in FEATS.items():
        X = feats 
        num_samples = len(X)
        
        # cross-validation -> probing method
        print("==> running probing...")
        X = scaler.fit_transform(X.reshape(num_samples,-1))
        cv_acc_list = []
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        for split_index, (train_index, test_index) in enumerate(cv.split(X, Y)):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = Y[train_index]
            y_test = Y[test_index]
            clf = LogisticRegression(max_iter=1200)
            clf.fit(X_train, y_train)
            # test 
            y_predict = clf.predict(X_test)
            _score = (y_predict == y_test).sum() / len(y_predict)
            cv_acc_list.append(_score)
        saved_dict[args.exp_name][f"{prob_name}_acc_{layer_name}"] = np.mean(np.array(cv_acc_list)) 
        
    return saved_dict
        
def compute_feature_weighting(args, saved_dict, FEATS):
    print("==> running feats weighting...")
    for layer_name, feats in FEATS.items():
        assert feats.shape[0] == 15000
        output_logit_ori = feats[:5000]
        output_logit_flip_fg = feats[5000:10000]
        output_logit_flip_bg = feats[10000:15000] 
    
        # compute diff
        flip_bg_diff_l1 = np.abs((output_logit_ori-output_logit_flip_bg)).mean()
        flip_fg_diff_l1 = np.abs((output_logit_ori-output_logit_flip_fg)).mean()
        flip_bg_diff_l2 = np.square((output_logit_ori-output_logit_flip_bg)).mean()
        flip_fg_diff_l2 = np.square((output_logit_ori-output_logit_flip_fg)).mean()
        
        # compute relative change
        saved_dict[args.exp_name][f'rel_change-rand_pair-l1-sub_{layer_name}'] = float(flip_fg_diff_l1 - flip_bg_diff_l1)
        saved_dict[args.exp_name][f'rel_change-rand_pair-l1-div_{layer_name}'] = float(flip_fg_diff_l1 / flip_bg_diff_l1)
        
        saved_dict[args.exp_name][f'rel_change-rand_pair-l2-sub_{layer_name}'] = float(flip_fg_diff_l2 - flip_bg_diff_l2)
        saved_dict[args.exp_name][f'rel_change-rand_pair-l2-div_{layer_name}'] = float(flip_fg_diff_l2 / flip_bg_diff_l2)
        
        if layer_name == 'output':
            from scipy.special import softmax
            sm = nn.Softmax(dim=1)
            kl = nn.KLDivLoss(reduction='batchmean')
            flip_bg_diff_kl = kl(sm(torch.tensor(output_logit_flip_bg)), 
                                 sm(torch.tensor(output_logit_ori)))
            flip_fg_diff_kl = kl(sm(torch.tensor(output_logit_flip_fg)), 
                                 sm(torch.tensor(output_logit_ori)))
        
            saved_dict[args.exp_name][f'rel_change-rand_pair-kl-sub_{layer_name}'] = float(flip_fg_diff_kl - flip_bg_diff_kl)        
            saved_dict[args.exp_name][f'rel_change-rand_pair-kl-div_{layer_name}'] = float(flip_fg_diff_kl / flip_bg_diff_kl)
            
    return saved_dict
        
    
    # # relative change - naive diff in activation
    # output_logit_list = []
    # for X, y, _mask_size in data_loader:
    #     X = X.float().cuda()
    #     output_logit = model.predict(X)
    #     output_logit_list.append(output_logit.detach().cpu() / _mask_size.unsqueeze(-1))
    # output_logit_list = torch.cat(output_logit_list).numpy()
    # act_scale = np.abs(output_logit_list).mean(1)
    # rel_act_scale = act_scale[:1000].mean() - act_scale[1000:].mean()
    # rel_act_scale_v2 = act_scale[:1000].mean() / act_scale[1000:].mean()
    # saved_dict[args.exp_name]["rel_act_scale"] = rel_act_scale  
    # saved_dict[args.exp_name]["rel_act_scale_v2"] = rel_act_scale_v2  
    
def compute_probing_trad_metrics(args, saved_dict, FEATS, Y_fg, Y_bg, prob_name):
    scaler = StandardScaler()
    for layer_name, feats in FEATS.items():
        X = feats 
        num_samples = len(X)
        
        # cross-validation -> probing method
        print("==> running classic probing...")
        X = scaler.fit_transform(X.reshape(num_samples,-1))
        cv_fg_acc_list = []
        cv_bg_acc_list = []
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        for split_index, (train_index, test_index) in enumerate(cv.split(X, Y_fg)):
            X_train = X[train_index]
            X_test = X[test_index]
            ## fg
            y_fg_train = Y_fg[train_index]
            y_fg_test = Y_fg[test_index]
            clf_fg = LogisticRegression(max_iter=10000)
            clf_fg.fit(X_train, y_fg_train)
            y_fg_predict = clf_fg.predict(X_test)
            _score = (y_fg_predict == y_fg_test).sum() / len(y_fg_predict)
            cv_fg_acc_list.append(_score)
            ## bg
            y_bg_train = Y_bg[train_index]
            y_bg_test = Y_bg[test_index]
            clf_bg = LogisticRegression(max_iter=10000)
            clf_bg.fit(X_train, y_bg_train)
            y_bg_predict = clf_bg.predict(X_test)
            _score = (y_bg_predict == y_bg_test).sum() / len(y_bg_predict)
            cv_bg_acc_list.append(_score)
        saved_dict[args.exp_name][f"{prob_name}_trad_fg_{layer_name}"] = np.mean(cv_fg_acc_list)
        saved_dict[args.exp_name][f"{prob_name}_trad_bg_{layer_name}"] = np.mean(cv_bg_acc_list)
        
    return saved_dict
        
def compute_probing_nSquare_metrics(args, saved_dict, FEATS, Y_fg, Y_bg, prob_name):
    scaler = StandardScaler()
    for layer_name, feats in FEATS.items():
        X = feats 
        Y = Y_fg * 10 + Y_bg
        num_samples = len(X)
        
        # cross-validation -> probing method
        print("==> running probing nSqaure...")
        X = scaler.fit_transform(X.reshape(num_samples,-1))
        cv_acc_list = []
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        for split_index, (train_index, test_index) in enumerate(cv.split(X, Y)):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = Y[train_index]
            y_test = Y[test_index]
            clf = LogisticRegression(max_iter=1200)
            clf.fit(X_train, y_train)
            # test 
            y_predict = clf.predict(X_test)
            _score = (y_predict == y_test).sum() / len(y_predict)
            cv_acc_list.append(_score)
        saved_dict[args.exp_name][f"{prob_name}_nSqaure_acc_{layer_name}"] = np.mean(np.array(cv_acc_list)) 
        
    return saved_dict


def compute_fact_subspaces(args, saved_dict, FEATS, num_center = "1000", is_record = True):
    print("computing fact subspaces...")
    for layer_name, feats in FEATS.items():
        print(f"==>{layer_name}...")
        dim = feats.shape[-1]
        num_rand_pairs = int(feats.shape[0] / 1000 / 2)
        if num_center == "1000":
            shape = (1000, num_rand_pairs, dim)
        elif num_center == "10":
            shape = (10, num_rand_pairs*100, dim)

        # get centers
        fg_feats = feats[:1000*num_rand_pairs].reshape(shape)
        bg_feats = feats[1000*num_rand_pairs:].reshape(shape)
        fg_centers = fg_feats.mean(1)
        bg_centers = bg_feats.mean(1)

        # compute var
        var_fg = np.var(bg_feats, axis=1).sum(1).mean() # var induced by fg perturbation
        var_bg = np.var(fg_feats, axis=1).sum(1).mean()
        var_all_fg = np.var(bg_feats.reshape((1000*num_rand_pairs,dim)), axis=0).sum()
        var_all_bg = np.var(fg_feats.reshape((1000*num_rand_pairs,dim)), axis=0).sum()

        # compute PCA space containing 99% of the variance
        pca_fg = PCA(n_components=0.99) 
        pca_fg.fit(fg_centers)
        pca_bg = PCA(n_components=0.99) 
        pca_bg.fit(bg_centers)

        # compute var in subspaces
        fg_feats_in_bg_subspace = pca_bg.transform(bg_feats.reshape(1000*num_rand_pairs, dim))
        fg_feats_in_bg_subspace = fg_feats_in_bg_subspace.reshape((shape[0], shape[1], -1))
        var_fg_in_bg_subspace = np.var(fg_feats_in_bg_subspace, axis=1).sum(1).mean()

        bg_feats_in_fg_subspace = pca_fg.transform(fg_feats.reshape(1000*num_rand_pairs, dim))
        bg_feats_in_fg_subspace = bg_feats_in_fg_subspace.reshape((shape[0], shape[1], -1))
        var_bg_in_fg_subspace = np.var(bg_feats_in_fg_subspace, axis=1).sum(1).mean()

        ## factorization metrics
        fac_fg = 1 - var_fg_in_bg_subspace / var_fg
        fac_bg = 1 - var_bg_in_fg_subspace / var_bg
        inv_fg = 1 - var_fg / var_all_fg
        inv_bg = 1 - var_bg / var_all_bg

        if is_record:
            saved_dict[args.exp_name][f'fact_subspaces_fg_samples{num_rand_pairs}_{layer_name}'] = fac_fg
            saved_dict[args.exp_name][f'fact_subspaces_bg_samples{num_rand_pairs}_{layer_name}'] = fac_bg
            saved_dict[args.exp_name][f'inv_subspaces_fg_samples{num_rand_pairs}_{layer_name}'] = inv_fg
            saved_dict[args.exp_name][f'inv_subspaces_bg_samples{num_rand_pairs}_{layer_name}'] = inv_bg
        else:
            print("factorization - fg: %.4f, bg: %.4f" % (fac_fg, fac_bg)) 
            print("invariance - fg: %.4f, bg: %.4f" % (inv_fg, inv_bg))
            return fac_fg, fac_bg, inv_fg, inv_bg
    
    return saved_dict


from sklearn.model_selection import StratifiedKFold

def cv_test(X, Y, n_splits=5, boost_factor=1):
    cv_acc_list = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for split_index, (train_index, test_index) in enumerate(cv.split(X, Y)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]
        clf = LogisticRegression(max_iter=10000, C=1/(boost_factor**2)) # , penalty='l2', C=1.0
        clf.fit(X_train, y_train)
        # test 
        y_predict = clf.predict(X_test)
        _score = (y_predict == y_test).sum() / len(y_predict)
        cv_acc_list.append(_score)
    print("acc: %.2f+-%.2f" % (np.mean(cv_acc_list)*100, np.std(cv_acc_list)*100))
    return np.mean(cv_acc_list), np.std(cv_acc_list)