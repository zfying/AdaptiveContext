# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np

def _hparams(algorithm, dataset, random_state, args):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}
    hparams['opt'] = (args.opt, args.opt)
    hparams['data_augmentation'] = (True, True)
    hparams['resnet50'] = (args.resnet50, False)
    hparams['resnet_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    hparams['class_balanced'] = (args.class_balanced, False)
    
    args.epochs = 100
    
    ## add algo-specific arguments
    ## MYCODE ##
    # general
    hparams['use_mask'] = (args.use_mask, None)
    hparams['use_two_labels'] = (args.use_two_labels, None)
    hparams['feat_ext'] = (args.feat_ext_list, None)
    hparams['sup_ratio'] = (args.sup_ratio, None)
    # probing loss
    hparams['probing_loss_weight'] = (args.probing_loss_weight, None)
    hparams['use_orthogonal_loss'] = (args.use_orthogonal_loss, None)
    hparams['orthogonal_loss_weight'] = (args.orthogonal_loss_weight, None)
    # RSA loss
    hparams['RSA_loss_type'] = (args.RSA_loss_type, None)
    hparams['RSA_loss_weight'] = (args.RSA_loss_weight, None)
    # augmentation
    hparams['aug_fg'] = (args.aug_fg, None)
    hparams['aug_fg_type'] = (args.aug_fg_type, None)
    hparams['aug_bg'] = (args.aug_bg, None)
    hparams['aug_bg_type'] = (args.aug_bg_type, None)
    hparams['fg_weight'] = (args.fg_weight, None)
    hparams['bg_weight'] = (args.bg_weight, None)
    # wt_evid
    hparams['wt_evid_loss_weight'] = (args.wt_evid_loss_weight, None)
    hparams['dist_metrics'] = (args.dist_metrics, None)
    hparams['comp_metrics'] = (args.comp_metrics, None)
    hparams['target_diff'] = (args.target_diff, None)
    hparams['detach'] = (args.detach, None)
    ## END MYCODE ##
    
    ### MYCODE ###
    if dataset == 'SceneCOCO':
        hparams['lr'] = (1e-1, 10**random_state.uniform(-5, -3.5)) # ORIGINAL
        hparams['batch_size'] = (64, int(2 ** random_state.uniform(3, 9)))
        hparams['sch_size'] = (1500, 100)
        # hparams['batch_size'] = (128, int(2 ** random_state.uniform(3, 9))) # ORIGINAL
        # hparams['lr'] = (5e-2, 10**random_state.uniform(-5, -3.5))
        # hparams['sch_size'] = (600, 100) # ORIGINAL
    elif dataset == 'ColorObject':
        hparams['lr'] = (1e-1, 10**random_state.uniform(-5, -3.5)) # ORIGINAL
        hparams['batch_size'] = (128, int(2 ** random_state.uniform(3, 9)))
        hparams['sch_size'] = (1500, 100)
    else:
        assert False
    ### MYCODE ###


    hparams['weight_decay'] = (1e-4, 10**random_state.uniform(-6, -2))

    if algorithm in ['DANN', 'CDANN']:
        if dataset not in SMALL_IMAGES:
            hparams['lr_g'] = (5e-5, 10**random_state.uniform(-5, -3.5))
            hparams['lr_d'] = (5e-5, 10**random_state.uniform(-5, -3.5))
        else:
            hparams['lr_g'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))
            hparams['lr_d'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))

        if dataset not in SMALL_IMAGES:
            hparams['weight_decay_g'] = (0., 10**random_state.uniform(-6, -2))
        # else:
        #     hparams['weight_decay_g'] = (0., 0.)

        hparams['lambda'] = (1.0, 10**random_state.uniform(-2, 2))
        hparams['weight_decay_d'] = (0., 10**random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2**random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0., 10**random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    elif algorithm == "RSC":
        hparams['rsc_f_drop_factor'] = (1/3, random_state.uniform(0,0.5)) # Feature drop factor
        hparams['rsc_b_drop_factor'] = (1/3, random_state.uniform(0, 0.5)) # Batch drop factor
    elif algorithm == 'Fish':
        hparams['meta_lr'] = (args.fish_lam, lambda r: r.choice([0.05, 0.1, 0.5]))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))
    elif algorithm == "SagNet":
        hparams['sag_w_adv'] = (0.1, 10**random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams['irm_lambda'] = (args.irm_lam, 10**random_state.uniform(-1, 5))
        hparams['irm_penalty_anneal_iters'] = (100, int(10**random_state.uniform(0, 4)))
    elif algorithm == "Mixup":
        hparams['mixup_alpha'] = (0.2, 10**random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams['groupdro_eta'] = (args.dro_eta, 10**random_state.uniform(-3, -1))
        hparams['weight_decay'] = (1e-4, 0.)
    elif algorithm == "MMD" or algorithm == "CORAL":
        hparams['mmd_gamma'] = (1., 10**random_state.uniform(-1, 1))
    elif algorithm == "MLDG":
        hparams['mldg_beta'] = (1, 10**random_state.uniform(-1, 1))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))
    elif algorithm == "MTL":
        hparams['mtl_ema'] = (.99, random_state.choice([0.5, 0.9, 0.99, 1.]))
    elif algorithm == "VREx":
        hparams['vrex_lambda'] = (args.rex_lam, 10**random_state.uniform(-1, 5))
        hparams['vrex_penalty_anneal_iters'] = (500, int(10**random_state.uniform(0, 4)))
    elif algorithm == "ERM":
        pass
    elif algorithm == "TRM":
        hparams['cos_lambda'] = (args.cos_lam, 10 ** random_state.uniform(-5, 0))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))
        hparams['groupdro_eta'] = (args.dro_eta, 10 ** random_state.uniform(-3, -1))

    return hparams

def default_hparams(algorithm, dataset, args):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, dummy_random_state, args).items()}

def random_hparams(algorithm, dataset, seed, args):
    random_state = np.random.RandomState(seed)
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, random_state, args).items()}
