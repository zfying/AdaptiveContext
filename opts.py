import argparse
from distutils.util import strtobool 

def parse_opt(s=None):
    parser = argparse.ArgumentParser()
    
    ## basics
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="SceneCOCO", choices=["SceneCOCO", "ColorObject"])
    parser.add_argument('--model_save', dest='model_save', 
                    type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--num_domains', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    
    
    ## task specific arguments
    parser.add_argument('--algorithm', type=str, required=True) # ["ERM", "IRM", "VREx", "GroupDRO", "Fish", "MLDG"]
                                                                # ["ERM_probing", ""ERM_RSA", "ERM_augmentation"]
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--bias', type=str, required=True)
    # dataset type
    parser.add_argument('--use_two_labels', action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    # general args
    parser.add_argument('--feat_ext_list', nargs='+', default=None)
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
    # parser.add_argument('--test_val', action='store_true', help="test-domain validation set")
    
    if s is not None:
        args = parser.parse_args("")
    else:
        args = parser.parse_args()

    return args