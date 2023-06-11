import numpy as np 
import PIL
import matplotlib.pyplot as plt
import torch
import pickle

import os, subprocess
import numpy as np
import random
from PIL import Image
from skimage.transform import resize
import matplotlib
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)

## read preprocessed coco and places data

h5_coco_train = h5py.File("../data/coco_train.h5py", "r")
coco_train_imgs = h5_coco_train['resized_images']
coco_train_masks = h5_coco_train['resized_mask']

h5_coco_val = h5py.File("../data/coco_validtest.h5py", "r")
coco_val_imgs = h5_coco_val['resized_images']
coco_val_masks = h5_coco_val['resized_mask']

h5_coco_test = h5py.File("../data/coco_idtest.h5py", "r")
coco_test_imgs = h5_coco_test['resized_images']
coco_test_masks = h5_coco_test['resized_mask']

h5_places = h5py.File("../data/places.h5py", "r")
places_imges = h5_places['resized_place']

### for SceneObject

## get train set
# init
for bias_coef in [0.7, 0.9]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/cocoplaces/train_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (8000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (8000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (8000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_indexes', (8000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 800

    for class_index in range(10):
        for image_index in tqdm(range(num_imgs_pre_class)):
            # get mask and img
            _mask = coco_train_masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = coco_train_imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            place_img_index = random.randint(0, num_imgs_pre_class-1)
            _place_img = places_imges[place_class_index][place_img_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["fg_indexes"][_index] = image_index # [0,799]
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["bg_indexes"][_index] = place_img_index
    h5_file.close()

## get val set
# init
for bias_coef in [0.7, 0.9]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/cocoplaces/val_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_indexes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 800
    masks = coco_val_masks
    imgs = coco_val_imgs

    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            place_img_index = random.randint(0, num_imgs_pre_class-1) + place_img_index_start
            _place_img = places_imges[place_class_index][place_img_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
            h5_file["bg_indexes"][_index] = place_img_index
    h5_file.close()

## get id test set
# init
for bias_coef in [0.7, 0.9]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/cocoplaces/test_id_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_indexes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 900
    masks = coco_test_masks
    imgs = coco_test_imgs


    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            place_img_index = random.randint(0, num_imgs_pre_class-1) + place_img_index_start
            _place_img = places_imges[place_class_index][place_img_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
            h5_file["bg_indexes"][_index] = place_img_index
    h5_file.close()

## get ood1 test set
# init
for bias_coef in [0.0, 0.15, 0.3, 0.45, 0.60]:  # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/cocoplaces/test_ood1_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_indexes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 900
    masks = coco_test_masks
    imgs = coco_test_imgs

    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            place_img_index = random.randint(0, num_imgs_pre_class-1) + place_img_index_start
            _place_img = places_imges[place_class_index][place_img_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
            h5_file["bg_indexes"][_index] = place_img_index
    h5_file.close()

## get ood2 test set using corrupted images
# init
fname = os.path.join("../data/cocoplaces/test_ood2.h5py")
if os.path.exists(fname): subprocess.call(['rm', fname])
h5_file = h5py.File(fname, mode='w')
# size: 15 corruption type * 5 degree * 1000 = 75,000
h5_file.create_dataset('images', (75000,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('fg_classes', (75000,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_classes', (75000,), dtype=np.dtype('int32'))
h5_file.create_dataset('fg_indexes', (75000,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_indexes', (75000,), dtype=np.dtype('int32'))
num_imgs_pre_class = 100
place_img_index_start = 900
masks = coco_test_masks

corruption_name_list = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 
                        'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 
                        'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
for c_type_index, corruption_name in enumerate(corruption_name_list):
    for c_degree_index, corruption_degree in enumerate(['1','2','3','4','5']):
        img_root = os.path.join('../data/coco/test_imgs_c/', corruption_name, corruption_degree, 'test_ori')
        for class_index in tqdm(range(10)):
            for image_index in range(num_imgs_pre_class):
                # read fg img
                index_string = str(class_index*num_imgs_pre_class + image_index).zfill(5)
                img_path = os.path.join(img_root, f'coco_test_{index_string}.png')
                _coco_img = Image.open(img_path)
                _coco_img = np.array(_coco_img).transpose([2,0,1]) / 255.
                # get fg mask 
                _mask = masks[class_index*num_imgs_pre_class + image_index]
                # get place img
                place_class_index = class_index
                place_img_index = random.randint(0, num_imgs_pre_class-1) + place_img_index_start
                _place_img = places_imges[place_class_index][place_img_index]

                # combine fg and bg
                _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

                # record
                _index = class_index*num_imgs_pre_class + image_index + 1000 * (c_degree_index+c_type_index*5)
                h5_file["images"][_index] = _combined_img
                h5_file["fg_classes"][_index] = class_index
                h5_file["bg_classes"][_index] = place_class_index
                h5_file["fg_indexes"][_index] = image_index 
                h5_file["bg_indexes"][_index] = place_img_index
h5_file.close()

# subset OOD2level
# read prob sets
for c_degree_index, corruption_degree in enumerate(['1','2','3','4','5']):
    h5_file_ori = h5py.File(f"../data/cocoplaces/test_ood2level{corruption_degree}.h5py", "r")

    subset_ratio = 0.1
    print(f"subset_ratio: {subset_ratio}")
    num_samples = int(15000 * subset_ratio)
    # init new file
    fname = os.path.join(f"../data/cocoplaces/test_ood2level{corruption_degree}_subset{subset_ratio}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (num_samples,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (num_samples,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (num_samples,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (num_samples,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_indexes', (num_samples,), dtype=np.dtype('int32'))

    all_index = np.random.permutation(15000)[:num_samples]
    for cur_index, ori_index in tqdm(enumerate(all_index)):
        h5_file["images"][cur_index] = h5_file_ori['images'][ori_index]
        h5_file["fg_classes"][cur_index] = h5_file_ori['fg_classes'][ori_index]
        h5_file["bg_classes"][cur_index] = h5_file_ori['bg_classes'][ori_index]
        h5_file["fg_indexes"][cur_index] = h5_file_ori['fg_indexes'][ori_index]
        h5_file["bg_indexes"][cur_index] = h5_file_ori['bg_indexes'][ori_index]
    h5_file.close()
    h5_file_ori.close()
    
    
## for subspaces 
num_images = 10
num_rand_paires = 100
fname = os.path.join(f"../data/cocoplaces/test_subspaces_sample{num_rand_paires}.h5py")
if os.path.exists(fname): subprocess.call(['rm', fname])
h5_file = h5py.File(fname, mode='w')
h5_file.create_dataset('images', (num_images*num_rand_paires*2,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('masks', (num_images*num_rand_paires*2,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('fg_classes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
h5_file.create_dataset('fg_indexes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_classes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_indexes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
num_imgs_pre_class = 100
place_img_index_start = 900
masks = coco_test_masks
imgs = coco_test_imgs

# fg + random background
for class_index in tqdm(range(10)):
    for image_index in range(num_imgs_pre_class):
        # get mask and img
        _mask = masks[class_index*num_imgs_pre_class + image_index]
        _coco_img = imgs[class_index*num_imgs_pre_class + image_index]
        for sample_index in range(num_rand_paires):
            # get random places
            place_class_index = random.randint(0, 9)
            place_img_index = random.randint(0, num_imgs_pre_class-1) + place_img_index_start
            _place_img = places_imges[place_class_index][place_img_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = (class_index*num_imgs_pre_class + image_index) * num_rand_paires + sample_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
            h5_file["bg_indexes"][_index] = place_img_index

# bg + random foreground
for class_index in tqdm(range(10)):
    for image_index in range(num_imgs_pre_class):
        # get places
        place_class_index = class_index
        place_img_index = image_index + place_img_index_start
        _place_img = places_imges[place_class_index][place_img_index]
        
        for sample_index in range(num_rand_paires):
            # get random mask and img
            fg_random_class_index = random.randint(0, 9)
            fg_random_img_index = random.randint(0, num_imgs_pre_class-1)
            _mask = masks[fg_random_class_index*num_imgs_pre_class + fg_random_img_index]
            _coco_img = imgs[fg_random_class_index*num_imgs_pre_class + fg_random_img_index]
            
            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = num_images*num_rand_paires + (class_index*num_imgs_pre_class + image_index) * num_rand_paires + sample_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = fg_random_class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = fg_random_img_index 
            h5_file["bg_indexes"][_index] = place_img_index
h5_file.close()



### for ColorObject
bg_color_imgs = []
bg_colors =  np.array([[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225], [0, 0, 255], [255, 20, 147],
[160, 160, 160]])
for bg_index in range(10):
    _img = np.ones((64,64,3)) * bg_colors[bg_index] / 255.
    bg_color_imgs.append(_img.transpose([2,0,1]))
    
    
## for subspaces 
# 1000 fg img * 10 random bg = 10k
# 1000 bg img * 10 random fg = 10k
num_images = 1000
num_rand_paires = 100
fname = os.path.join(f"../data/colorobject/test_subspaces_sample{num_rand_paires}.h5py")
if os.path.exists(fname): subprocess.call(['rm', fname])
h5_file = h5py.File(fname, mode='w')
h5_file.create_dataset('images', (num_images*num_rand_paires*2,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('masks', (num_images*num_rand_paires*2,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('fg_classes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
h5_file.create_dataset('fg_indexes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_classes', (num_images*num_rand_paires*2,), dtype=np.dtype('int32'))

num_imgs_pre_class = 100
masks = coco_test_masks
imgs = coco_test_imgs

# fg + random background
for class_index in tqdm(range(10)):
    for image_index in range(num_imgs_pre_class):
        # get mask and img
        _mask = masks[class_index*num_imgs_pre_class + image_index]
        _coco_img = imgs[class_index*num_imgs_pre_class + image_index]
        for sample_index in range(num_rand_paires):
            # get random places
            place_class_index = random.randint(0, 9)
            _place_img = bg_color_imgs[place_class_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = (class_index*num_imgs_pre_class + image_index) * num_rand_paires + sample_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 

# bg + random foreground
for class_index in tqdm(range(10)):
    # get places
    place_class_index = class_index
    _place_img = bg_color_imgs[place_class_index]        
        
    for sample_index in range(num_rand_paires*num_imgs_pre_class):
        # get random mask and img
        fg_random_class_index = random.randint(0, 9)
        fg_random_img_index = random.randint(0, num_imgs_pre_class-1)
        _mask = masks[fg_random_class_index*num_imgs_pre_class + fg_random_img_index]
        _coco_img = imgs[fg_random_class_index*num_imgs_pre_class + fg_random_img_index]

        # combine fg and bg
        _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

        # record
        _index = num_images*num_rand_paires + class_index * (num_rand_paires*num_imgs_pre_class) + sample_index
        h5_file["images"][_index] = _combined_img
        h5_file["masks"][_index] = _mask
        h5_file["fg_classes"][_index] = fg_random_class_index
        h5_file["bg_classes"][_index] = place_class_index
        h5_file["fg_indexes"][_index] = fg_random_img_index 
            
h5_file.close()


## get train set
# init
for bias_coef in [0.7, 1.0]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/colorobject/train_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (8000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (8000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (8000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 800

    for class_index in range(10):
        for image_index in tqdm(range(num_imgs_pre_class)):
            # get mask and img
            _mask = coco_train_masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = coco_train_imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            _place_img = bg_color_imgs[place_class_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["fg_indexes"][_index] = image_index # [0,799]
            h5_file["bg_classes"][_index] = place_class_index
    h5_file.close()
    
## get val set
# init
for bias_coef in [0.7, 1.0]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/colorobject/val_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 800
    masks = coco_val_masks
    imgs = coco_val_imgs

    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            _place_img = bg_color_imgs[place_class_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
    h5_file.close()
    
## get id test set
# init
for bias_coef in [0.7, 1.0]: # obey relationship with prob, o.w. random
    fname = os.path.join(f"../data/colorobject/test_id_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (8000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 900
    masks = coco_test_masks
    imgs = coco_test_imgs


    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            _place_img = bg_color_imgs[place_class_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
    h5_file.close()
    
## get ood1 test set
for bias_coef in [0.0, 0.15, 0.3, 0.45, 0.60]:
    # init
    fname = os.path.join(f"../data/colorobject/test_ood1_{bias_coef}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('masks', (1000,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (1000,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (1000,), dtype=np.dtype('int32'))
    num_imgs_pre_class = 100
    place_img_index_start = 900
    masks = coco_test_masks
    imgs = coco_test_imgs

    for class_index in tqdm(range(10)):
        for image_index in range(num_imgs_pre_class):
            # get mask and img
            _mask = masks[class_index*num_imgs_pre_class + image_index]
            _coco_img = imgs[class_index*num_imgs_pre_class + image_index]

            # get place according to bias coef
            if random.uniform(0, 1) < bias_coef: 
                # sample bg from corresponding class
                place_class_index = class_index
            else:
                # randomly sample place class
                place_class_index = random.randint(0, 9)
            _place_img = bg_color_imgs[place_class_index]

            # combine fg and bg
            _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

            # record
            _index = class_index*num_imgs_pre_class + image_index
            h5_file["images"][_index] = _combined_img
            h5_file["masks"][_index] = _mask
            h5_file["fg_classes"][_index] = class_index
            h5_file["bg_classes"][_index] = place_class_index
            h5_file["fg_indexes"][_index] = image_index 
    h5_file.close()
    
## get ood2 test set using corrupted images
# init
fname = os.path.join("../data/colorobject/test_ood2.h5py")
if os.path.exists(fname): subprocess.call(['rm', fname])
h5_file = h5py.File(fname, mode='w')
# size: 15 corruption type * 5 degree * 1000 = 75,000
h5_file.create_dataset('images', (75000,3,64,64), dtype=np.dtype('float32'))
h5_file.create_dataset('fg_classes', (75000,), dtype=np.dtype('int32'))
h5_file.create_dataset('bg_classes', (75000,), dtype=np.dtype('int32'))
h5_file.create_dataset('fg_indexes', (75000,), dtype=np.dtype('int32'))
num_imgs_pre_class = 100
place_img_index_start = 900
masks = coco_test_masks

corruption_name_list = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 
                        'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 
                        'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
for c_type_index, corruption_name in tqdm(enumerate(corruption_name_list)):
    for c_degree_index, corruption_degree in enumerate(['1','2','3','4','5']):
        img_root = os.path.join('../data/context-ori-data/test_imgs_c/', corruption_name, corruption_degree, 'test_ori')
        for class_index in range(10):
            for image_index in range(num_imgs_pre_class):
                # read fg img
                index_string = str(class_index*num_imgs_pre_class + image_index).zfill(5)
                img_path = os.path.join(img_root, f'coco_test_{index_string}.png')
                _coco_img = Image.open(img_path)
                _coco_img = np.array(_coco_img).transpose([2,0,1]) / 255.
                # get fg mask 
                _mask = masks[class_index*num_imgs_pre_class + image_index]
                # get place img
                place_class_index = class_index
                _place_img = bg_color_imgs[place_class_index]

                # combine fg and bg
                _combined_img  = _coco_img * _mask + _place_img * (1-_mask)

                # record
                _index = class_index*num_imgs_pre_class + image_index + 1000 * (c_degree_index+c_type_index*5)
                h5_file["images"][_index] = _combined_img
                h5_file["fg_classes"][_index] = class_index
                h5_file["bg_classes"][_index] = place_class_index
                h5_file["fg_indexes"][_index] = image_index 
h5_file.close()

# subset OOD2level
# read prob sets
for c_degree_index, corruption_degree in enumerate(['1','2','3','4','5']):
    h5_file_ori = h5py.File(f"../data/colorobject/test_ood2level{corruption_degree}.h5py", "r")

    subset_ratio = 0.1
    print(f"subset_ratio: {subset_ratio}")
    num_samples = int(15000 * subset_ratio)
    # init new file
    fname = os.path.join(f"../data/colorobject/test_ood2level{corruption_degree}_subset{subset_ratio}.h5py")
    if os.path.exists(fname): subprocess.call(['rm', fname])
    h5_file = h5py.File(fname, mode='w')
    h5_file.create_dataset('images', (num_samples,3,64,64), dtype=np.dtype('float32'))
    h5_file.create_dataset('fg_classes', (num_samples,), dtype=np.dtype('int32'))
    h5_file.create_dataset('bg_classes', (num_samples,), dtype=np.dtype('int32'))
    h5_file.create_dataset('fg_indexes', (num_samples,), dtype=np.dtype('int32'))

    all_index = np.random.permutation(15000)[:num_samples]
    for cur_index, ori_index in tqdm(enumerate(all_index)):
        h5_file["images"][cur_index] = h5_file_ori['images'][ori_index]
        h5_file["fg_classes"][cur_index] = h5_file_ori['fg_classes'][ori_index]
        h5_file["bg_classes"][cur_index] = h5_file_ori['bg_classes'][ori_index]
        h5_file["fg_indexes"][cur_index] = h5_file_ori['fg_indexes'][ori_index]
    h5_file.close()
    h5_file_ori.close()
    
