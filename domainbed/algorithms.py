# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# 'ERM', 'VREx', 'GroupDRO' are the same with one domain?
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import copy
import numpy as np
from domainbed.inv_hvp import neum, sto
from domainbed import networks
from domainbed import grad_fun
from domainbed.lib.misc import random_pairs_of_minibatches, ParamDict
from domainbed.lib.misc import kl_loss_function
import torch.nn.utils.prune as prune

from pytorch_metric_learning import losses


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


## 10*10=100-class probing
class ERM_probing(Algorithm):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_probing, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        ## MYCODE ##
        self.feat_ext = self.hparams['feat_ext']
        assert(self.feat_ext is not None)
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        self.probing_loss_weight = self.hparams['probing_loss_weight']
        self.use_orthogonal_loss = self.hparams['use_orthogonal_loss']
        assert(self.use_orthogonal_loss==False)
        self.orthogonal_loss_weight = self.hparams['orthogonal_loss_weight']
        self.sup_ratio = self.hparams['sup_ratio']
        self.probing_classifiers = []
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = None

    def update(self, minibatches, is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        all_mask = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
            all_mask.append(mask)
        # label
        all_y0 = torch.cat(all_y0)
        all_y1 = torch.cat(all_y1)
        cur_batch_size = len(all_y0)
        # input
        all_img = torch.cat(all_img)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_mask = torch.cat(all_mask)
        # create fg + random bg & bg + random fg
        random_order_0 = torch.randperm(cur_batch_size)
        all_x_new_0 = all_fg_only * all_mask + all_bg_only[random_order_0] * (1-all_mask)
        all_y_new_0 = all_y0 * self.num_classes + all_y1[random_order_0] # 10*10 = 100 classes
        random_order_1 = torch.randperm(cur_batch_size)
        all_x_new_1 = all_fg_only[random_order_1] * all_mask + all_bg_only * (1-all_mask)
        all_y_new_1 = all_y0[random_order_1] * self.num_classes + all_y1 # 10*10 = 100 classes
        # final input & output
        all_x = torch.cat([all_img, all_x_new_0, all_x_new_1], dim=0)
        all_y_new = torch.cat([all_y_new_0, all_y_new_1], dim=0)

        ## forward
        feature, add_feat = self.featurizer(all_x, self.feat_ext)
        ## task loss
        loss = F.cross_entropy(self.classifier(feature[:cur_batch_size]), all_y0)
        ## probing loss
        loss_add = 0
        num_ext_layers = len(add_feat)
        for feat_index, ext_feats in enumerate(add_feat): 
            ext_feats_new = ext_feats[cur_batch_size:].view(cur_batch_size*2, -1)
            if len(self.probing_classifiers) != len(add_feat):
                self.probing_classifiers.append(nn.Linear(ext_feats_new.shape[1], self.num_classes * self.num_classes).cuda())
            pred = self.probing_classifiers[feat_index](ext_feats_new)
            mask = (torch.cuda.FloatTensor(pred.shape[0]).uniform_() < self.sup_ratio)
            if mask.sum() != 0:
                loss_add += F.cross_entropy(pred[mask], all_y_new[mask])
        loss_add = loss_add / num_ext_layers* self.probing_loss_weight
        loss = loss + loss_add 
        
        # ## orthogonal loss
        # if self.use_orthogonal_loss:
        #     loss_orth = 0
        #     for layer_index in range(num_ext_layers):
        #         for i in range(self.num_classes):
        #             for j in range(self.num_classes):
        #                 loss_orth += torch.dot(self.probing_classifiers[layer_index].weight[i], \
        #                                        self.probing_classifiers[layer_index].weight[j+self.num_classes])
        #     loss = loss + loss_orth * self.orthogonal_loss_weight
        
        # init optim
        if self.optimizer is None: 
            # init optimizer & scheduler
            if self.hparams['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )
            elif self.hparams['opt'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)
        
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {'loss': float(loss.item()), 'loss_add': float(loss_add.item())}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()


## two 10-class probing
class ERM_probing_2heads(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_probing_2heads, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        ## MYCODE ##
        self.feat_ext = self.hparams['feat_ext']
        assert(self.feat_ext is not None)
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        self.probing_loss_weight = self.hparams['probing_loss_weight']
        self.use_orthogonal_loss = self.hparams['use_orthogonal_loss']
        self.orthogonal_loss_weight = self.hparams['orthogonal_loss_weight']
        self.sup_ratio = self.hparams['sup_ratio']
        self.fg_classifiers = []
        self.bg_classifiers = []
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = None

    def update(self, minibatches, is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
        # label
        all_y0 = torch.cat(all_y0)
        all_y1 = torch.cat(all_y1)
        cur_batch_size = len(all_y0)
        # input
        all_img = torch.cat(all_img)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_x = torch.cat([all_img, all_fg_only, all_bg_only], dim=0)

        ## forward
        feature, add_feat = self.featurizer(all_x, self.feat_ext)
        ## task loss
        loss = F.cross_entropy(self.classifier(feature[:cur_batch_size]), all_y0)
        ## probing loss
        loss_add = 0
        num_ext_layers = len(add_feat)
        for feat_index, ext_feats in enumerate(add_feat): 
            ext_feats_fg = ext_feats[cur_batch_size:cur_batch_size*2].view(cur_batch_size, -1)
            ext_feats_bg = ext_feats[cur_batch_size*2:cur_batch_size*3].view(cur_batch_size, -1)
            assert(ext_feats_fg.shape == ext_feats_bg.shape)
            if len(self.fg_classifiers) != len(add_feat):
                self.fg_classifiers.append(nn.Linear(ext_feats_fg.shape[1], self.num_classes).cuda())
                self.bg_classifiers.append(nn.Linear(ext_feats_fg.shape[1], self.num_classes).cuda())
            loss_add += F.cross_entropy(self.fg_classifiers[feat_index](ext_feats_fg), all_y0)
            loss_add += F.cross_entropy(self.bg_classifiers[feat_index](ext_feats_bg), all_y1)
        loss_add /= num_ext_layers*2
        loss = loss + loss_add * self.probing_loss_weight

        ## orthogonal loss
        if self.use_orthogonal_loss:
            loss_orth = 0
            for layer_index in range(num_ext_layers):
                for i in range(self.num_classes):
                    for j in range(self.num_classes):
                        loss_orth += torch.dot(self.probing_classifiers[feat_index].weight[i], \
                                               self.probing_classifiers[feat_index].weight[j+self.num_classes])
            loss = loss + loss_orth * self.orthogonal_loss_weight
            
        # init optim
        if self.optimizer is None: 
            # init optimizer & scheduler
            if self.hparams['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )
            elif self.hparams['opt'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {'loss': float(loss.item())}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()

        
## 10+10=20-class probing
class ERM_probing_old(Algorithm):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_probing_old, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        ## MYCODE ##
        self.feat_ext = self.hparams['feat_ext']
        assert(self.feat_ext is not None)
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        self.probing_loss_weight = self.hparams['probing_loss_weight']
        self.use_orthogonal_loss = self.hparams['use_orthogonal_loss']
        self.orthogonal_loss_weight = self.hparams['orthogonal_loss_weight']
        self.sup_ratio = self.hparams['sup_ratio']
        self.probing_classifiers = []
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = None

    def update(self, minibatches,  is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
        # label
        all_y0 = torch.cat(all_y0)
        all_y1 = torch.cat(all_y1)
        cur_batch_size = len(all_y0)
        all_y = torch.cat([all_y0, all_y1+self.num_classes], dim=0)
        # input
        all_img = torch.cat(all_img)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_x = torch.cat([all_img, all_fg_only, all_bg_only], dim=0)

        ## forward
        feature, add_feat = self.featurizer(all_x, self.feat_ext)
        ## task loss
        loss = F.cross_entropy(self.classifier(feature[:cur_batch_size]), all_y[:cur_batch_size])
        ## probing loss
        loss_add = 0
        num_ext_layers = len(add_feat)
        for feat_index, ext_feats in enumerate(add_feat): 
            ext_feats_fg = ext_feats[cur_batch_size:cur_batch_size*2].view(cur_batch_size, -1)
            ext_feats_bg = ext_feats[cur_batch_size*2:cur_batch_size*3].view(cur_batch_size, -1)
            assert(ext_feats_fg.shape == ext_feats_bg.shape)
            if len(self.probing_classifiers) != len(add_feat):
                self.probing_classifiers.append(nn.Linear(ext_feats_fg.shape[1], self.num_classes * 2).cuda())
            pred = self.probing_classifiers[feat_index](torch.cat([ext_feats_fg, ext_feats_bg], dim=0))
            mask = (torch.cuda.FloatTensor(pred.shape[0]).uniform_() < self.sup_ratio)
            if mask.sum() != 0:
                loss_add += F.cross_entropy(pred[mask], all_y[mask])
        loss_add /= num_ext_layers
        loss = loss + loss_add * self.probing_loss_weight
        
        ## orthogonal loss
        if self.use_orthogonal_loss:
            loss_orth = 0
            for layer_index in range(num_ext_layers):
                for i in range(self.num_classes):
                    for j in range(self.num_classes):
                        loss_orth += torch.dot(self.probing_classifiers[layer_index].weight[i], \
                                               self.probing_classifiers[layer_index].weight[j+self.num_classes])
            loss = loss + loss_orth * self.orthogonal_loss_weight
        
        # init optim
        if self.optimizer is None: 
            # init optimizer & scheduler
            if self.hparams['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )
            elif self.hparams['opt'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)
        
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {'loss': float(loss.item())}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()

        
## RSA loss -> contrastive loss
class ERM_RSA(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_RSA, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        ## MYCODE ##
        self.feat_ext = self.hparams['feat_ext']
        assert(self.feat_ext is not None)
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        self.RSA_loss_type = self.hparams['RSA_loss_type']
        self.RSA_loss_weight = self.hparams['RSA_loss_weight']
        
        if self.RSA_loss_type == 'contrastive_loss':
            self.loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        else:
            assert False, f"unsupported RSA loss type: {self.RSA_loss_type}"
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = None

    def update(self, minibatches,  is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
        # label
        all_y0 = torch.cat(all_y0)
        all_y1 = torch.cat(all_y1)
        # input
        all_img = torch.cat(all_img)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_x = torch.cat([all_img, all_fg_only, all_bg_only], dim=0)
        cur_batch_size = len(all_y0)
        
        ## forward
        feature, add_feat = self.featurizer(all_x, self.feat_ext)
        ## task loss
        loss = F.cross_entropy(self.classifier(feature[:cur_batch_size]), all_y0)
        
        ## RSA loss
        loss_add = 0
        num_ext_layers = len(add_feat)
        for feat_index, ext_feats in enumerate(add_feat): 
            ext_feats = ext_feats[cur_batch_size:].view(cur_batch_size*2, -1) # fg&bg rep
            all_y = torch.cat([all_y0, all_y1+self.num_classes])
            loss_add = self.loss_func(ext_feats, all_y)
        loss = loss + loss_add * self.RSA_loss_weight
            
        # optim
        if self.optimizer is None: 
            # init optimizer & scheduler
            if self.hparams['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )
            elif self.hparams['opt'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)
        
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()

        
## augmentation
class ERM_augmentation(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_augmentation, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        ## MYCODE ##
        self.hparams = hparams
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        self.aug_fg = self.hparams['aug_fg']
        self.aug_bg = self.hparams['aug_bg']
        self.aug_fg_type = self.hparams['aug_fg_type']
        self.aug_bg_type = self.hparams['aug_bg_type']
        self.fg_weight = self.hparams['fg_weight']
        self.bg_weight = self.hparams['bg_weight']
        assert(self.fg_weight + self.bg_weight <=1.0)
        assert(self.aug_fg or self.aug_bg)
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        # init optimizer & scheduler
        if self.hparams['opt'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    def update(self, minibatches, is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        all_mask = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
            all_mask.append(mask)
        # label
        all_y0 = torch.cat(all_y0)
        all_y1 = torch.cat(all_y1)
        cur_batch_size = len(all_y0)
        all_y = [all_y0]
        # input
        all_img = torch.cat(all_img)
        all_mask = torch.cat(all_mask)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_x = [all_img]
        if self.aug_fg:
            # add x
            if self.aug_fg_type == 'fg_only':
                all_x.append(all_fg_only)
            elif self.aug_fg_type == 'random_bg':
                # random bg
                all_fg_random = all_bg_only[torch.randperm(cur_batch_size)] * (1-all_mask) + all_fg_only * all_mask
                all_x.append(all_fg_random)
            else:
                assert False, f"unsupported fg aug: {self.aug_fg_type}"
            # add y
            all_y.append(all_y0)
        if self.aug_bg:
            # add x
            if self.aug_bg_type == 'bg_only':
                all_x.append(all_bg_only)
            else:
                assert False  , f"unsupported bg aug: {self.aug_bg_type}"
            # add y
            all_y.append(all_y1)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        if self.aug_fg and self.aug_bg: # three batches -> apply fg_weight & bg_weight
            assert all_x.shape[0] == cur_batch_size * 3
            feature = self.featurizer(all_x)
            all_y_pred = self.classifier(feature)
            # ori loss
            loss_task = F.cross_entropy(all_y_pred[:cur_batch_size], all_y[:cur_batch_size]) * (1-self.fg_weight-self.bg_weight)
            loss_fg_aug = F.cross_entropy(all_y_pred[cur_batch_size:cur_batch_size*2], 
                                          all_y[cur_batch_size:cur_batch_size*2]) * self.fg_weight
            loss_bg_aug = F.cross_entropy(all_y_pred[cur_batch_size*2:cur_batch_size*3], 
                                          all_y[cur_batch_size*2:cur_batch_size*3]) * self.bg_weight
            loss = loss_task + loss_fg_aug  + loss_bg_aug
        else: # two batches -> assume equal weights
            ## forward
            feature = self.featurizer(all_x)
            ## task loss
            loss = F.cross_entropy(self.classifier(feature), all_y)
        
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        if self.aug_fg and self.aug_bg:
            return {'loss': loss.item(), 'loss_task': loss_task.item(), 
                    'loss_fg_aug': loss_fg_aug.item(), 'loss_bg_aug': loss_bg_aug.item()}
        else:
            return {'loss': loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()
        

## weighted evidence loss
class ERM_wt_evid(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_wt_evid, self).__init__(input_shape, num_classes, num_domains, hparams)
        ## MYCODE ##
        self.hparams = hparams
        assert(self.hparams['feat_ext'] is None)
        assert(self.hparams['use_mask'])
        assert(self.hparams['use_two_labels'])
        self.num_classes = num_classes
        # opts for wt_evid
        self.wt_evid_loss_weight = self.hparams['wt_evid_loss_weight']
        self.dist_metrics = self.hparams['dist_metrics']
        self.comp_metrics = self.hparams['comp_metrics']
        self.target_diff = self.hparams['target_diff']
        self.detach = self.hparams['detach']
        ## END MYCODE ##
        
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        self.optimizer = None
        
    def update(self, minibatches, is_update=True, add_loss=True):
        # init
        all_img = []
        all_y0 = []
        all_y1 = []
        all_fg_only = []
        all_bg_only = []
        all_mask = []
        for x, (y0, y1), mask in minibatches:
            # label 
            all_y0.append(y0)
            all_y1.append(y1)
            # input
            img = x
            fg_only = x * mask
            bg_only = x * (1-mask)
            all_img.append(img)
            all_fg_only.append(fg_only)
            all_bg_only.append(bg_only)
            all_mask.append(mask)
        # label
        all_y0 = torch.cat(all_y0) # fg labels
        all_y1 = torch.cat(all_y1) # bg labels
        cur_batch_size = len(all_y0)
        all_y = [all_y0]
        # input
        all_img = torch.cat(all_img)
        all_mask = torch.cat(all_mask)
        all_fg_only = torch.cat(all_fg_only)
        all_bg_only = torch.cat(all_bg_only)
        all_x = [all_img]
        # second batch - same fg, random bg
        rand_index = torch.randperm(cur_batch_size)
        all_fg_random_bg = all_bg_only[rand_index] * (1-all_mask) + all_fg_only * all_mask
        all_x.append(all_fg_random_bg)
        all_y.append(all_y0)
        # third batch - random fg, same bg
        rand_index = torch.randperm(cur_batch_size)
        all_bg_random_fg = all_fg_only[rand_index] * all_mask + all_bg_only * (1-all_mask)
        all_x.append(all_bg_random_fg)
        all_y.append(all_y0[rand_index])
        # final inputs/labels
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        ## forward
        feature = self.featurizer(all_x)
        ## task loss
        loss = F.cross_entropy(self.classifier(feature[:cur_batch_size]), all_y0)
        
        ## wt_evid loss
        if add_loss: # add_loss -> support adding loss after some epochs
            # get output logits
            if self.detach: # detach & train only the classifier
                output_logits = self.classifier(feature.detach()) 
            else:
                output_logits = self.classifier(feature) 
            # resize extracted features
            ext_feats_ori = output_logits[:cur_batch_size].view(cur_batch_size, -1)
            ext_feats_flip_bg = output_logits[cur_batch_size : cur_batch_size*2].view(cur_batch_size, -1)
            ext_feats_flip_fg = output_logits[cur_batch_size*2 : cur_batch_size*3].view(cur_batch_size, -1)
            # compute diff
            if self.dist_metrics == 'l1':
                flip_bg_diff = (ext_feats_flip_bg-ext_feats_ori).abs().mean()
                flip_fg_diff = (ext_feats_flip_fg-ext_feats_ori).abs().mean()
            elif self.dist_metrics == 'l2':
                flip_bg_diff = (ext_feats_flip_bg-ext_feats_ori).square().mean()
                flip_fg_diff = (ext_feats_flip_fg-ext_feats_ori).square().mean()
            elif self.dist_metrics == 'kl':
                sm = nn.Softmax(dim=1)
                kl = nn.KLDivLoss()
                flip_bg_diff = kl(sm(ext_feats_flip_bg), sm(ext_feats_ori))
                flip_fg_diff = kl(sm(ext_feats_flip_fg), sm(ext_feats_ori))
            # compute relative change
            if self.comp_metrics == 'sub':
                rel_diff = flip_fg_diff - flip_bg_diff
            elif self.comp_metrics == 'div':
                rel_diff = flip_fg_diff / flip_bg_diff
            loss_add = (self.target_diff - rel_diff).square() * self.wt_evid_loss_weight
            loss += loss_add
        else:
            loss_add = 0
            
        # init optim
        if self.optimizer is None: 
            # init optimizer & scheduler
            if self.hparams['opt'] == 'SGD':
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )
            elif self.hparams['opt'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)
        
        # optim
        loss.backward()
        if is_update:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {'loss': float(loss.item()), 'loss_add': float(loss_add)}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    def predict_feature(self, x):
        return self.featurizer(x)
    def predict_classifier(self, feature):
        return self.classifier(feature)
    def train(self):
        self.featurizer.train()
    def eval(self):
        self.featurizer.eval()
        
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # init network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(4)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(4)]
        # init optimizer & scheduler
        if self.hparams['opt'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, is_update=True, add_loss=True):
        # get data
        all_x = []
        all_y = []
        for x, y in minibatches:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)
        # forward
        feature = self.featurizer(all_x)
        loss = F.cross_entropy(self.classifier(feature), all_y)
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler.step()
        return {'loss': float(loss.item())}

    def predict(self, x):
        return self.network(x)

    def predict_feature(self, x):
        return self.featurizer(x)

    def predict_classifier(self, feature):
        return self.classifier(feature)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


class EC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(EC, self).__init__(input_shape, num_classes, num_domains,
                                 hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.latent_dim = 64
        self.input_dim = self.featurizer.n_outputs
        self.num_domains = num_domains
        del self.featurizer

        self.classifiers = [torch.nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), nn.ReLU(True),
                                                nn.Linear(self.latent_dim, num_classes)).cuda() for i in
                            range(num_domains)]
        self.optimizer = [torch.optim.Adam(
            self.classifiers[i].parameters(),
            lr=1e-2,
            weight_decay=self.hparams['weight_decay']
        ) for i in range(num_domains)]

    def update_ec(self, minibatches, feature):
        features = feature.detach()
        start = 0
        for i in range(self.num_domains):
            loss = F.cross_entropy(self.classifiers[i](features[start: start + minibatches[i][1].size(0)]),
                                   minibatches[i][1])
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            start += minibatches[i][1].size(0)

    def predict_envs(self, env, x):
        return self.classifiers[env](x)

    def predict(self, x):
        pass

class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)

class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, is_update=True, add_loss=True):
        penalty_weight = (
            self.hparams['irm_lambda'] if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 0)

        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        self.network.train()
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            loss_ = F.cross_entropy(logits, y)
            nll += loss_
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        if self.update_count == self.hparams['irm_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam' and  self.hparams['irm_lambda'] != 1:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    def update(self, minibatches, is_update=True, add_loss=True):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 0.

        nll = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = (mean + penalty_weight * penalty)

        if self.update_count == self.hparams['vrex_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam':
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler.step()
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                       hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, is_update=True, add_loss=True):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        # print(losses)
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler.step()

        return {'loss': loss.item()}

class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.update_count = 0

    def update(self, minibatches, unlabeled=None, is_update=True, add_loss=True):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)
            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)
            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)
        That is, when calling .step(), we want grads to be Gi + beta * Gj
        For computational efficiency, we do not compute second derivatives.
        """
        if self.update_count < self.hparams['iters']:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            # updating original network
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # updating scheduler (only for SceneCOCO and C-MNIST)
            self.scheduler.step()
            self.update_count += 1
            return {'loss': loss.item()}

        
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"

            inner_net = copy.deepcopy(self.network)
            if self.hparams["opt"] == 'Adam':
                inner_opt = torch.optim.Adam(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            else:
                inner_opt = torch.optim.SGD(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )

            # updating original network
            inner_obj = F.cross_entropy(inner_net(xi), yi)
            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # update bn
            F.cross_entropy(self.network(xi), yi)
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()
            # self.optimizer.load_state_dict(inner_opt.state_dict())
            # # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()
        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler.step()
        self.update_count += 1
        return {'loss': objective}


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = networks.Featurizer(input_shape, hparams)
        classifier = nn.Linear(featurizer.n_outputs, num_classes)
        self.net = nn.Sequential(
            featurizer, classifier
        )
        self.featurizer = featurizer
        self.classifier = classifier
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
    
    def predict_feature(self, x):
        return self.featurizer(x)
        

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = WholeFish(input_shape, num_classes, hparams)
        self.optimizer_inner_state = None
        self.step = 0
        if self.hparams['opt'] == 'SGD':
            self.optimizer_inner = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_inner = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
    def create_clone(self, device):
        self.network_inner = WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)

        if self.step > 0 and self.step % self.hparams['sch_size'] == 0:
            self.hparams["lr"] *= 0.1

        if self.hparams['opt'] == 'SGD':
            self.optimizer_inner = torch.optim.SGD(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_inner = torch.optim.Adam(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

        for param_group in self.optimizer_inner.param_groups:
            param_group['lr'] = self.hparams["lr"]

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None, is_update=True, add_loss=True):

        self.create_clone(minibatches[0][0].device)
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)
        self.step += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def predict_feature(self, x):
        return self.network.predict_feature(x)
    
class TRM(Algorithm):
    """
    TRM
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,
                                          hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(4)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(4)]

        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_f = torch.optim.Adam(
                self.featurizer.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler_f = torch.optim.lr_scheduler.StepLR(self.optimizer_f, step_size=self.hparams['sch_size'],gamma=0.1)
        self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=self.hparams['sch_size'], gamma=0.1)
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()


    def update(self, minibatches):

        loss_swap = 0.0
        trm = 0.0
        # updating featurizer
        if self.update_count >= self.hparams['iters']:
            # if self.hparams['class_balanced']:
            # for stability when facing unbalanced labels across environments
            for classifier in self.clist:
                classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            # print(loss_swap, self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q))
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        #if self.update_count >= self.hparams['iters']:
        if self.update_count >= 300:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # updating scheduler (only for SceneCOCO and C-MNIST)
        self.scheduler_f.step()
        self.scheduler_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

        
# class AbstractDANN(Algorithm):
#     """Domain-Adversarial Neural Networks (abstract class)"""

#     def __init__(self, input_shape, num_classes, num_domains,
#                  hparams, conditional, class_balance):

#         super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
#                                            hparams)

#         self.register_buffer('update_count', torch.tensor([0]))
#         self.conditional = conditional
#         self.class_balance = class_balance

#         # Algorithms
#         self.featurizer = networks.Featurizer(input_shape, self.hparams)
#         self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
#         self.discriminator = networks.MLP(self.featurizer.n_outputs,
#                                           num_domains, self.hparams)
#         self.class_embeddings = nn.Embedding(num_classes,
#                                              self.featurizer.n_outputs)

#         # Optimizers
#         if self.hparams["opt"] == 'Adam':
#             self.disc_opt = torch.optim.Adam(
#                 (list(self.discriminator.parameters()) +
#                 list(self.class_embeddings.parameters())),
#                 lr=self.hparams["lr_d"],
#                 weight_decay=self.hparams['weight_decay_d'],
#                 betas=(self.hparams['beta1'], 0.9))

#             self.gen_opt = torch.optim.Adam(
#                 (list(self.featurizer.parameters()) +
#                 list(self.classifier.parameters())),
#                 lr=self.hparams["lr_g"],
#                 weight_decay=self.hparams['weight_decay_g'],
#                 betas=(self.hparams['beta1'], 0.9))
#         else:
#             self.disc_opt = torch.optim.SGD(
#                 (list(self.discriminator.parameters()) +
#                  list(self.class_embeddings.parameters())),
#                 lr=self.hparams["lr"],
#                 momentum=0.9,
#                 weight_decay=self.hparams['weight_decay']
#             )
#             self.gen_opt = torch.optim.SGD(
#                 (list(self.featurizer.parameters()) +
#                 list(self.classifier.parameters())),
#                 lr=self.hparams["lr"],
#                 momentum=0.9,
#                 weight_decay=self.hparams['weight_decay']
#             )
#     def update(self, minibatches):
#         self.update_count += 1
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])
#         all_z = self.featurizer(all_x)
#         if self.conditional:
#             disc_input = all_z + self.class_embeddings(all_y)
#         else:
#             disc_input = all_z
#         disc_out = self.discriminator(disc_input)
#         disc_labels = torch.cat([
#             torch.full((x.shape[0],), i, dtype=torch.int64, device='cuda')
#             for i, (x, y) in enumerate(minibatches)
#         ])

#         if self.class_balance:
#             y_counts = F.one_hot(all_y).sum(dim=0)
#             weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
#             disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
#             disc_loss = (weights * disc_loss).sum()
#         else:
#             disc_loss = F.cross_entropy(disc_out, disc_labels)

#         disc_softmax = F.softmax(disc_out, dim=1)
#         input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
#                                    [disc_input], create_graph=True)[0]
#         grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
#         disc_loss += self.hparams['grad_penalty'] * grad_penalty

#         d_steps_per_g = self.hparams['d_steps_per_g_step']
#         if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

#             self.disc_opt.zero_grad()
#             disc_loss.backward()
#             self.disc_opt.step()
#             return {'disc_loss': disc_loss.item()}
#         else:
#             all_preds = self.classifier(all_z)
#             classifier_loss = F.cross_entropy(all_preds, all_y)
#             gen_loss = (classifier_loss +
#                         (self.hparams['lambda'] * -disc_loss))
#             self.disc_opt.zero_grad()
#             self.gen_opt.zero_grad()
#             gen_loss.backward()
#             self.gen_opt.step()
#             return {'gen_loss': gen_loss.item()}

#     def predict(self, x):
#         return self.classifier(self.featurizer(x))

# class DANN(AbstractDANN):
#     """Unconditional DANN"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(DANN, self).__init__(input_shape, num_classes, num_domains,
#                                    hparams, conditional=False, class_balance=False)

# class CDANN(AbstractDANN):
#     """Conditional DANN"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CDANN, self).__init__(input_shape, num_classes, num_domains,
#                                     hparams, conditional=True, class_balance=True)


# class AbstractMMD(ERM):
#     """
#     Perform ERM while matching the pair-wise domain feature distributions
#     using MMD (abstract class)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
#         super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
#                                           hparams)
#         if gaussian:
#             self.kernel_type = "gaussian"
#         else:
#             self.kernel_type = "mean_cov"

#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                            1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y):
#         if self.kernel_type == "gaussian":
#             Kxx = self.gaussian_kernel(x, x).mean()
#             Kyy = self.gaussian_kernel(y, y).mean()
#             Kxy = self.gaussian_kernel(x, y).mean()
#             return Kxx + Kyy - 2 * Kxy
#         else:
#             mean_x = x.mean(0, keepdim=True)
#             mean_y = y.mean(0, keepdim=True)
#             cent_x = x - mean_x
#             cent_y = y - mean_y
#             cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
#             cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

#             mean_diff = (mean_x - mean_y).pow(2).mean()
#             cova_diff = (cova_x - cova_y).pow(2).mean()

#             return mean_diff + cova_diff

#     def update(self, minibatches, unlabeled=None):
#         objective = 0
#         penalty = 0
#         # domain number
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
#         classifs = [self.classifier(fi) for fi in features]
#         targets = [yi for _, yi in minibatches]

#         for i in range(nmb):
#             objective += F.cross_entropy(classifs[i], targets[i])
#             for j in range(i + 1, nmb):
#                 penalty += self.mmd(features[i], features[j])

#         objective /= nmb
#         if nmb > 1:
#             penalty /= (nmb * (nmb - 1) / 2)

#         self.optimizer.zero_grad()
#         (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
#         self.optimizer.step()
#         # updating scheduler (only for SceneCOCO and C-MNIST)
#         self.scheduler.step()
#         if torch.is_tensor(penalty):
#             penalty = penalty.item()

#         return {'loss': objective.item(), 'penalty': penalty}


# class MMD(AbstractMMD):
#     """
#     MMD using Gaussian kernel
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(MMD, self).__init__(input_shape, num_classes,
#                                           num_domains, hparams, gaussian=True)


# class CORAL(AbstractMMD):
#     """
#     MMD using mean and covariance difference
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CORAL, self).__init__(input_shape, num_classes,
#                                          num_domains, hparams, gaussian=False)


# class network_two_stream(nn.Module):
#     """Just  an MLP"""
#     def __init__(self, input_shape, num_classes, hparams):
#         super(network_two_stream, self).__init__()
#         self.featurizer = networks.Featurizer(input_shape, hparams)
#         self.classifier_0 = nn.Linear(self.featurizer.n_outputs, num_classes)
#         self.classifier_1 = nn.Linear(self.featurizer.n_outputs, num_classes)

#     def forward(self, x):
#         x = self.featurizer(x)
#         y0 = self.classifier_0(x)
#         y1 = self.classifier_0(x)
#         return y0, y1
        
# class ERM_two_stream(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(ERM_two_stream, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         # init network
#         self.network = network_two_stream(input_shape, num_classes, hparams)
#         # init optimizer & scheduler
#         if self.hparams['opt'] == 'SGD':
#             self.optimizer = torch.optim.SGD(
#                 self.network.parameters(),
#                 lr=self.hparams["lr"],
#                 momentum=0.9,
#                 weight_decay=self.hparams['weight_decay']
#             )
#             # self.optimizer_c = torch.optim.SGD(
#             #     self.network.classifier.parameters(),
#             #     lr=self.hparams["lr"],
#             #     momentum=0.9,
#             #     weight_decay=self.hparams['weight_decay']
#             # )
#         elif self.hparams['opt'] == 'Adam':
#             self.optimizer = torch.optim.Adam(
#                 self.network.parameters(),
#                 lr=self.hparams["lr"],
#                 weight_decay=self.hparams['weight_decay']
#             )
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

#     def update(self, minibatches):
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y0 = torch.cat([y0 for x, (y0, y1) in minibatches])
#         all_y1 = torch.cat([y1 for x, (y0, y1) in minibatches])

#         output = self.network(all_x)
#         loss0 = F.cross_entropy(output[0], all_y0)
#         loss1 = F.cross_entropy(output[1], all_y1)
#         loss = (loss0 + loss1) / 2
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         # updating scheduler (only for SceneCOCO and C-MNIST)
#         self.scheduler.step()

#         return {'loss': loss.item()}

#     def predict(self, x):
#         return self.network(x)

#     def predict_feature(self, x):
#         return self.featurizer(x)

#     def predict_classifier(self, feature):
#         return self.classifier(feature)

#     def train(self):
#         self.network.train()

#     def eval(self):
#         self.network.eval()