import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F

import h5py

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
import opts

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, use_two_labels, use_mask):
        super(SimpleDataset, self).__init__()
        self.h5_file = h5py.File(os.path.join(data_root, split+".h5py"), "r")
        print(f"dataset size for split {split}: {len(self.h5_file['images'])}")
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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


args = opts.parse_opt()
args.num_classes = 10
args.input_shape = (3, 64, 64,)

# sanity check
if args.dataset == 'SceneCOCO':
    assert args.data_dir == "../data/cocoplaces", f"wrong data root for SceneCOCO dataset: {args.data_dir}"
elif args.dataset == 'ColorObject':
    assert args.data_dir == "../data/colorobject", f"wrong data root for ColorObject dataset: {args.data_dir}"
else:
    assert False, f"unsupported dataset: {args.dataset}"

def main(epoch):
    global last_results_keys
    global best_acc_out
    global best_acc_in
    global collect_dict
    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = zip(*train_loaders)

    for batch_idx, batch in enumerate(train_minibatches_iterator):
        step_start_time = time.time()
        if args.use_two_labels and args.use_mask:
            minibatches_device = [(x.cuda(), (y0.long().cuda(), y1.long().cuda()), mask.cuda())
                                  for x, (y0, y1), mask in batch]
        elif args.use_two_labels:
            minibatches_device = [(x.cuda(), (y0.long().cuda(), y1.long().cuda()))
                                  for x, (y0, y1) in batch]
        elif args.use_mask:
            minibatches_device = [(x.cuda(), y0.long().cuda(), mask.cuda())
                                  for x, y0, mask in batch]
        else:
            minibatches_device = [(x.cuda(), y0.long().cuda())
                                  for x, y0 in batch]
        
        
        # add loss late!
        if args.algorithm == 'ERM_wt_evid':
            if args.delay_adding_loss:
                assert(args.delay_epoch is not None)
                if epoch>=args.delay_epoch:
                    add_loss = True
                else: 
                    add_loss = False
            else:
                add_loss = True
        else:
            add_loss = False
            
        # gradient accumulation for large batches
        is_update = True
        if args.algorithm not in EXPANSIVE_ALGO or batch_size <= MAX_BATCH_SIZE: # simple algo OR small batch
            step_vals = algorithm.update(minibatches_device, True, add_loss)
        else: # expansive and large batch
            if (batch_idx + 1) % accum_iter == 0:
                step_vals = algorithm.update(minibatches_device, True, add_loss)
            else:
                step_vals = algorithm.update(minibatches_device, False, add_loss)
                is_update = False
        
        if is_update:
            checkpoint_vals['step_time'].append((time.time() - step_start_time)*accum_iter)
            args.step += 1

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)
    
    # test
    results = {
        'step': args.step,
        'epoch': epoch,
    }
    # test 
    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)
    evals = zip(eval_names, eval_loaders)
    for name, loader in evals:
        weights = None
        acc = misc.accuracy(algorithm, loader, weights, "cuda", args)                
        results[name + '_acc'] = acc

    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys],
                   colwidth=12)

    ## MYCODE ##
    val_acc = (results['val_0_acc'] + results['val_1_acc']) / 2
    test_acc = (results['test_0_acc'] + results['test_1_acc']) / 2
    # val_acc = results['val_0_acc']
    # test_acc = results['test_0_acc']
    ## END MYCODE ##

    collect_dict['acc'].append(test_acc)
    if val_acc > best_acc_in:
        best_acc_in = val_acc
        best_acc_out = test_acc
        path = os.path.join(args.output_dir, f"{args.exp_name}.pth")
        print(f"epoch:{epoch}, Val acc:{best_acc_in:.4f}, Test acc:{best_acc_out:.4f} ")
        print('----------------------------------')
        if args.model_save:
            save_dict = {
                "args": vars(args),
                "model_input_shape": args.input_shape,
                "model_num_classes": args.num_classes,
                "model_num_domains": args.num_domains,
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict()
            }
            print("save at...", path)
            torch.save(save_dict, path)
                    


args.step = 0
# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
algorithm_dict = None

os.makedirs(args.output_dir, exist_ok=True)
# sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
# sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

# read hyper params
if "ERM" in args.algorithm:
    hparams = hparams_registry.default_hparams("ERM", args.dataset, args)
else:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args)
if args.hparams:
    hparams.update(json.loads(args.hparams))
if args.lr is not None:
    hparams['lr'] = args.lr

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))
    

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## init dataset
# get max batch size for gradient accumulation
MAX_BATCH_SIZE = 64
EXPANSIVE_ALGO = ['ERM_probing', 'ERM_RSA', 'ERM_augmentation', 'ERM_probing_2heads', 'ERM_wt_evid']
if hparams['batch_size']>MAX_BATCH_SIZE and args.algorithm in EXPANSIVE_ALGO: # expansive algo AND large batch
    batch_size = MAX_BATCH_SIZE
    assert(hparams['batch_size'] % MAX_BATCH_SIZE == 0) # only support evenly divisible batch size
    accum_iter = int(hparams['batch_size']/MAX_BATCH_SIZE)
else:
    batch_size = hparams['batch_size']
    accum_iter = 1

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
    
train_0_dataset = SimpleDataset(args.data_dir, f'train_{bias_0}', args.use_two_labels, args.use_mask)
train_1_dataset = SimpleDataset(args.data_dir, f'train_{bias_1}', args.use_two_labels, args.use_mask)

train_0_loader = torch.utils.data.DataLoader(dataset=train_0_dataset, batch_size=batch_size,
                                         num_workers=args.num_workers, shuffle=True)
train_1_loader = torch.utils.data.DataLoader(dataset=train_1_dataset, batch_size=batch_size,
                                         num_workers=args.num_workers, shuffle=True)

val_0_dataset = SimpleDataset(args.data_dir, f'val_{bias_0}', args.use_two_labels, args.use_mask)
val_1_dataset = SimpleDataset(args.data_dir, f'val_{bias_1}', args.use_two_labels, args.use_mask)
test_id_0_dataset = SimpleDataset(args.data_dir, f'test_id_{bias_0}', args.use_two_labels, args.use_mask)
test_id_1_dataset = SimpleDataset(args.data_dir, f'test_id_{bias_1}', args.use_two_labels, args.use_mask)
test_ood1_dataset = SimpleDataset(args.data_dir, 'test_ood1', args.use_two_labels, args.use_mask)

val_0_loader = torch.utils.data.DataLoader(dataset=val_0_dataset, batch_size=hparams['batch_size'],
                                         num_workers=args.num_workers, shuffle=False)
val_1_loader = torch.utils.data.DataLoader(dataset=val_1_dataset, batch_size=hparams['batch_size'],
                                         num_workers=args.num_workers, shuffle=False)
test_id_0_loader = torch.utils.data.DataLoader(dataset=test_id_0_dataset, batch_size=hparams['batch_size'],
                                         num_workers=args.num_workers, shuffle=False)
test_id_1_loader = torch.utils.data.DataLoader(dataset=test_id_1_dataset, batch_size=hparams['batch_size'],
                                         num_workers=args.num_workers, shuffle=False)
test_ood1_loader = torch.utils.data.DataLoader(dataset=test_ood1_dataset, batch_size=hparams['batch_size'],
                                         num_workers=args.num_workers, shuffle=False)

train_loaders = [train_0_loader, train_1_loader]
eval_loaders = [val_0_loader, val_1_loader, test_id_0_loader, test_id_1_loader, test_ood1_loader]
eval_names = ['val_0', 'val_1', 'test_0', 'test_1', 'test_ood1']

## get alg
print("loading model...")
algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(args.input_shape, args.num_classes,
                            args.num_domains, hparams)
if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.resume_path}.pth')
    algorithm.load_state_dict(checkpoint['model_dict'], strict=False)

if args.parallel:
    algorithm = torch.nn.DataParallel(algorithm)
algorithm.cuda()
steps_per_epoch = len(train_0_dataset) / hparams['batch_size']
print("steps per epoch:", steps_per_epoch)

if args.dataset == 'SceneCOCO':
    args.epochs = int(4000. / steps_per_epoch)
elif args.dataset == 'ColorObject':
    args.epochs = int(4000. / steps_per_epoch)
else:
    assert False

checkpoint_freq = args.checkpoint_freq
last_results_keys = None
best_acc_in = 0.
best_acc_out = 0.

collect_dict = collections.defaultdict(lambda: [])

# main loop
print("Training epochs:", args.epochs)
start_time = time.time()
for epoch in range(args.epochs):
    main(epoch)
print(f"seed:{args.seed}, " + 
      f"alg:{args.algorithm}, Best val acc:{best_acc_in:.4f}, Best test acc:{best_acc_out:.4f}")
print(f"total time: {start_time - time.time()}")