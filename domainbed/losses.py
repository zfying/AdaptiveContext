import torch


def fg_bg_probing_loss(feats, all_y_fg, all_y_bg):
    