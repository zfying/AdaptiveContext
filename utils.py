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

import metrics

def load_param(dataset):
    parser = argparse.ArgumentParser()
    ## basics
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str, default="SceneCOCO")
    parser.add_argument('--model_save', dest='model_save', 
                    type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)


    ## task specific arguments
    parser.add_argument('--algorithm', type=str) # ["ERM", "IRM", "VREx", "GroupDRO", "Fish", "MLDG"]
                                                                # ["ERM_probing", ""ERM_RSA", "ERM_augmentation"]
    parser.add_argument('--seed', type=int,
                        help='Seed for everything else')
    # dataset type
    parser.add_argument('--use_two_labels', action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    # general args
    parser.add_argument('--feat_ext_list', nargs='+', default=['layer3'])
    parser.add_argument('--sup_ratio', type=float, default=1.0)
    # probing loss
    parser.add_argument('--probing_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_orthogonal_loss', action='store_true')
    parser.add_argument('--orthogonal_loss_weight', type=float, default=1.0)
    # RSA loss
    parser.add_argument('--RSA_loss_type', type=str, default="contrastive_loss")
    parser.add_argument('--RSA_loss_weight', type=float, default=1.0)
    # augmentation
    parser.add_argument('--aug_fg', action='store_true')
    parser.add_argument('--aug_fg_type', type=str, default="fg_only", choices=["fg_only", "random_bg"])
    parser.add_argument('--aug_bg', action='store_true')
    parser.add_argument('--aug_bg_type', type=str, default="bg_only", choices=["bg_only"])
    parser.add_argument('--fg_weight', type=float, default=0.0)
    parser.add_argument('--bg_weight', type=float, default=0.0)
    # weighted evidence
    parser.add_argument('--wt_evid_loss_weight', type=float, default=0.1)
    parser.add_argument('--target_diff', type=float, default=10.0)
    parser.add_argument('--dist_metrics', type=str, default="l2", choices=["l2", "l1", "kl"])
    parser.add_argument('--comp_metrics', type=str, default="sub", choices=["sub", "div"])
    parser.add_argument('--delay_adding_loss', action='store_true')
    parser.add_argument('--delay_epoch', type=int, default=None)
    parser.add_argument('--detach', action='store_true')


    parser.add_argument('--opt', type=str, default="SGD")
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')

    # parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--save_path', type=str, default="model")
    parser.add_argument('--resume_path', default="model", type=str, help='path to resume the checkpoint')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # parser.add_argument('--bias', type=float, default=0.9, help='bias degree for ColoredMNIST and SceneCOCO')
    parser.add_argument('--irm_lam', type=float, default=1, help='IRM parameter')
    parser.add_argument('--rex_lam', type=float, default=1, help='VREx parameter')
    parser.add_argument('--cos_lam', type=float, default=1e-4, help='TRM parameter')
    parser.add_argument('--fish_lam', type=float, default=0.5, help='Fish parameter')
    parser.add_argument('--dro_eta', type=float, default=1e-2)    
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resnet50', action='store_true')
    parser.add_argument('--class_balanced', action='store_true', help="re-weight the classes")

    args = parser.parse_args("")
    
    # default params
    args.num_classes = 10
    args.input_shape = (3, 64, 64,)
    
    # dataset specific params
    if dataset == "SceneCOCO":
        args.data_dir = '../data/cocoplaces/'
        args.output_dir = './saved_models'
        args.dataset = 'SceneCOCO'
        args.bias = '9and7'
    elif dataset == "ColorObject":
        args.data_dir = '../data/colorobject'
        args.output_dir = './saved_models_color'
        args.dataset = 'ColorObject'
        args.bias = '10and7'
    else:
        assert False, f"unsupported dataset: {dataset}"
    
    return args

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, use_two_labels, use_mask):
        super(SimpleDataset, self).__init__()
        self.h5_file = h5py.File(os.path.join(data_root, split+".h5py"), "r")
        # print(f"dataset size for split {split}: {len(self.h5_file['images'])}")
        self.use_two_labels = use_two_labels
        self.use_mask = use_mask
        
    def __getitem__(self, index):
        x = self.h5_file['images'][index]
        # y = (self.h5_file['bg_classes'][index], self.h5_file['fg_classes'][index])
        y_fg = self.h5_file['fg_classes'][index]
        if self.use_two_labels and self.use_mask:
            y_bg = self.h5_file['bg_classes'][index]
            mask = self.h5_file['masks'][index]
            return x, (y_fg, y_bg), mask
        elif self.use_two_labels:
            y_bg = self.h5_file['bg_classes'][index]
            return x, (y_fg, y_bg)
        elif self.use_mask:
            _mask = self.h5_file['masks'][index]
            return x, y_fg, _mask
        else:
            return x, y_fg

    def __len__(self):
        return len(self.h5_file['images'])

class SimpleDataset_masksize(torch.utils.data.Dataset):
    def __init__(self, X, y, mask_size):
        super(SimpleDataset_masksize, self).__init__()
        self.X = X
        self.y = y
        self.mask_size = mask_size
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.mask_size[index]
    def __len__(self):
        return len(self.X)
    
def init_data_model(args, eval_names, print_verbose=False):
    args.step = 0

    os.makedirs(args.output_dir, exist_ok=True)
    # sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    # sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # read hyper params
    if "ERM" in args.algorithm:
        hparams = hparams_registry.default_hparams("ERM", args.dataset, args)
    else:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args)
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.lr is not None:
        hparams['lr'] = args.lr

    if print_verbose:
        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

    random.seed(args.trial_seed)
    np.random.seed(args.trial_seed)
    torch.manual_seed(args.trial_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ## init dataset
    # set bias term
    if args.bias == '9and7':
        bias_0 = 0.9
        bias_1 = 0.7
    elif args.bias == '8and6':
        bias_0 = 0.8
        bias_1 = 0.6
    elif args.bias == '10and7':
        bias_0 = 1.0
        bias_1 = 0.7
    else:
        assert False, f"unsupported bias: {args.bias}"
    # ID
    test_id_0_dataset = SimpleDataset(args.data_dir, f'test_id_{bias_0}', False, False)
    test_id_1_dataset = SimpleDataset(args.data_dir, f'test_id_{bias_1}', False, False)
    # OOD1
    test_ood1_0_dataset = SimpleDataset(args.data_dir, 'test_ood1_0.0', False, False)
    test_ood1_015_dataset = SimpleDataset(args.data_dir, 'test_ood1_0.15', False, False)
    test_ood1_03_dataset = SimpleDataset(args.data_dir, 'test_ood1_0.3', False, False)
    test_ood1_045_dataset = SimpleDataset(args.data_dir, 'test_ood1_0.45', False, False)
    test_ood1_06_dataset = SimpleDataset(args.data_dir, 'test_ood1_0.6', False, False)
    # OOD2
    test_ood2_level1_dataset = SimpleDataset(args.data_dir, 'test_ood2level1_subset0.1', False, False)
    test_ood2_level2_dataset = SimpleDataset(args.data_dir, 'test_ood2level2_subset0.1', False, False)
    test_ood2_level3_dataset = SimpleDataset(args.data_dir, 'test_ood2level3_subset0.1', False, False)
    test_ood2_level4_dataset = SimpleDataset(args.data_dir, 'test_ood2level4_subset0.1', False, False)
    test_ood2_level5_dataset = SimpleDataset(args.data_dir, 'test_ood2level5_subset0.1', False, False)
    test_ood2ext_dataset = SimpleDataset(args.data_dir, 'test_ood2ext', False, False)

    hparams['batch_size'] = 1024
    # ID
    test_id_0_loader = torch.utils.data.DataLoader(dataset=test_id_0_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_id_1_loader = torch.utils.data.DataLoader(dataset=test_id_1_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    # OOD1
    test_ood1_0_loader = torch.utils.data.DataLoader(dataset=test_ood1_0_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood1_015_loader = torch.utils.data.DataLoader(dataset=test_ood1_015_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood1_03_loader = torch.utils.data.DataLoader(dataset=test_ood1_03_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood1_045_loader = torch.utils.data.DataLoader(dataset=test_ood1_045_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood1_06_loader = torch.utils.data.DataLoader(dataset=test_ood1_06_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    # OOD2
    test_ood2_level1_loader = torch.utils.data.DataLoader(dataset=test_ood2_level1_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood2_level2_loader = torch.utils.data.DataLoader(dataset=test_ood2_level2_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood2_level3_loader = torch.utils.data.DataLoader(dataset=test_ood2_level3_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood2_level4_loader = torch.utils.data.DataLoader(dataset=test_ood2_level4_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood2_level5_loader = torch.utils.data.DataLoader(dataset=test_ood2_level5_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    test_ood2ext_loader = torch.utils.data.DataLoader(dataset=test_ood2ext_dataset, batch_size=hparams['batch_size'],
                                             num_workers=args.num_workers, shuffle=False)
    
    # test_ood2_loader = torch.utils.data.DataLoader(dataset=test_ood2_dataset, batch_size=hparams['batch_size'],
    #                                          num_workers=args.num_workers, shuffle=False)
    # test_ood2_05_loader = torch.utils.data.DataLoader(dataset=test_ood2_05_dataset, batch_size=hparams['batch_size'],
    #                                          num_workers=args.num_workers, shuffle=False)
    # test_ood2_01_loader = torch.utils.data.DataLoader(dataset=test_ood2_01_dataset, batch_size=hparams['batch_size'],
    #                                          num_workers=args.num_workers, shuffle=False)
    # test_ood2_001_loader = torch.utils.data.DataLoader(dataset=test_ood2_001_dataset, batch_size=hparams['batch_size'],
    #                                          num_workers=args.num_workers, shuffle=False)
    names2loaders = {'test_id_0': test_id_0_loader,
                    'test_id_1': test_id_1_loader,
                    'test_ood1_0.0': test_ood1_0_loader,
                    'test_ood1_0.15': test_ood1_015_loader,
                     'test_ood1_0.3': test_ood1_03_loader,
                     'test_ood1_0.45': test_ood1_045_loader,
                     'test_ood1_0.6': test_ood1_06_loader,
                     'test_ood2_level1': test_ood2_level1_loader,
                     'test_ood2_level2': test_ood2_level2_loader,
                     'test_ood2_level3': test_ood2_level3_loader,
                     'test_ood2_level4': test_ood2_level4_loader,
                     'test_ood2_level5': test_ood2_level5_loader,
                    'ood2ext': test_ood2ext_loader}
    eval_loaders = []
    for name in eval_names:
        assert name in names2loaders, f"{name} is not supported"
        eval_loaders.append(names2loaders[name])
    
    if print_verbose:
        print("loading model...")
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args.input_shape, args.num_classes,
                                args.num_domains, hparams)

    # Load checkpoint.
    assert os.path.isdir(args.output_dir), 'Error: no checkpoint directory found!'
    ### MYCODE ###
    _path = os.path.join(args.output_dir, f"{args.exp_name}.pth")
    ### END MYCODE ###
    if print_verbose:
        print(f'==> Resuming from checkpoint: {_path}')
    checkpoint = torch.load(_path)
    if args.algorithm == 'GroupDRO':
        algorithm.q = torch.ones(args.num_domains)
    algorithm.load_state_dict(checkpoint['model_dict'], strict=False)
    algorithm.cuda()
    algorithm.eval()
    
    return eval_loaders, algorithm

# 'test_id_0', 'test_id_1', 'ood1', 'ood2', 'ood2ext', 'ood2_severe'
def run_test(args, eval_names, saved_dict, print_verbose=False):
    # args.exp_name = _exp_name
    # args.trial_seed = _seed
    # args.seed = _seed
    if args.exp_name not in saved_dict:
        saved_dict[args.exp_name] = {}
    saved_dict[args.exp_name]['seed'] = args.seed
    saved_dict[args.exp_name]['algo'] = args.algorithm
    saved_dict[args.exp_name]['use_two_labels'] = args.use_two_labels
    saved_dict[args.exp_name]['use_mask'] = args.use_mask
    print(args.exp_name)

    start_time = time.time()
    
    # init data & model
    eval_loaders, algorithm = init_data_model(args, eval_names, print_verbose) 

    # main loop
    ###### NO NEED FOR LABELS OR MASK FOR TESTING!
    ori_use_two_labels = args.use_two_labels
    ori_use_mask = args.use_mask
    args.use_two_labels = False
    args.use_mask = False
    if print_verbose:
        print("testing...")
    evals = zip(eval_names, eval_loaders)
    results = {}
    for name, loader in evals:
        weights = None
        ### MYCODE ###
        acc = misc.accuracy(algorithm, loader, weights, "cuda", args)
        ### END MYCODE ###
        saved_dict[args.exp_name][name + '_acc'] = acc
        results[name + '_acc'] = acc
    
    if print_verbose:
        results_keys = sorted(results.keys())
        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys],
                       colwidth=12)
    print("time: %.2f" % (time.time() - start_time))
    
    args.use_two_labels = ori_use_two_labels
    args.use_mask = ori_use_mask
    
    return saved_dict


def get_feats(args, model, data_loader, layer_list, apply_pca=False):
    print("==> getting feats...")
    # add hooks
    activation = {}
    def getActivation(name, reg="output"):
        # the hook signature
        def hook(model, inputs, outputs):
            if reg=="output":
                activation[name] = outputs.detach()
            else:
                # import ipdb
                # ipdb.set_trace()
                activation[name] = inputs[0].detach()
        return hook
    hooks = []
    if args.algorithm != 'Fish':
        _featurizer = model.featurizer
        _classifier = model.classifier
    else: # for Fish
        _featurizer = model.network.net[0]
        _classifier = model.network.net[1]
    
    if 'layer1' in layer_list:
        hooks.append(_featurizer.layer1.register_forward_hook(getActivation("layer1")))
    if 'layer2' in layer_list:
        hooks.append(_featurizer.layer2.register_forward_hook(getActivation("layer2")))
    if 'layer3' in layer_list:
        # hooks.append(_featurizer.layer3.register_forward_hook(getActivation("layer3")))
        hooks.append(_classifier.register_forward_hook(getActivation("layer4", "input")))
    if 'output' in layer_list:
        hooks.append(_classifier.register_forward_hook(getActivation("output")))
    
    # get feats
    FEATS = {}
    with torch.no_grad():
        for X, y, _ in data_loader:
            X = X.float().cuda()
            _ = model.predict(X)
            for key, value in activation.items():
                if key not in FEATS:
                    FEATS[key] = []
                FEATS[key].append(value.cpu())
    for key, value in FEATS.items():
        print(f"got feats for {key}")
        new_value = torch.cat(value, dim=0).numpy()
        new_value = new_value.reshape((new_value.shape[0], -1))
        FEATS[key] = new_value
        if apply_pca and (key == 'layer1' or key == 'layer2'):
            print(f"apply pca to {key}")
            pca = PCA(n_components = 128)
            FEATS[key] = pca.fit_transform(new_value.reshape((new_value.shape[0], -1)))
    for hook_handle in hooks:
        hook_handle.remove()
    return FEATS


def prob_model(args, saved_dict, target_rdms, dataloaders, ylabels, print_verbose = False):
    data_loader_prob3, data_loader_rand_paires, data_loader_OOD1, data_loader_subspaces = dataloaders
    y_prob3, y_OOD1_fg, y_OOD1_bg = ylabels

    print(args.exp_name)
    start_time = time.time()

    eval_names = []
    _, model = init_data_model(args, eval_names)
    
    ### factorization - subspaces
    layer_list = ['layer1', 'layer2', 'layer3']
    # layer_list = ['layer3']
    FEATS = get_feats(args, model, data_loader_subspaces, layer_list, apply_pca=False)
    saved_dict = metrics.compute_fact_subspaces(args, saved_dict, FEATS)
    
#     ### relative change - random sampling
#     layer_list = ['layer1', 'layer2', 'layer3', 'output']
#     # layer_list = ['output']
#     FEATS = get_feats(args, model, data_loader_rand_paires, layer_list, apply_pca=False)
#     saved_dict = metrics.compute_feature_weighting(args, saved_dict, FEATS)
    
#     ### get feats for prob3
#     layer_list = ['layer1', 'layer2', 'layer3']
#     # layer_list = ['layer3']
#     FEATS = get_feats(args, model, data_loader_prob3, layer_list, apply_pca=True)
    
#     ### compute RSA metrics
#     saved_dict = metrics.compute_RSA_metrics(args, saved_dict, target_rdms, FEATS, "prob3")
    
#     ### compute probing metrics
#     saved_dict = metrics.compute_probing_metrics(args, saved_dict, FEATS, y_prob3, "prob3")
    
    # ### compute probing traditional / nSquare
    # layer_list = ['layer1', 'layer2', 'layer3']
    # # layer_list = ['layer3']
    # FEATS = get_feats(args, model, data_loader_OOD1, layer_list, apply_pca=True)
    # saved_dict = metrics.compute_probing_trad_metrics(args, saved_dict, FEATS, y_OOD1_fg, y_OOD1_bg, "probOOD1")
    # saved_dict = metrics.compute_probing_nSquare_metrics(args, saved_dict, FEATS, y_OOD1_fg, y_OOD1_bg, "probOOD1")
    
    print("time: %.4f" % (time.time() - start_time))
    
    return saved_dict