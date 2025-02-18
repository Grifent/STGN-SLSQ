### Process the SLSQ data into format required by STGN
# (excuse the terrible code, half was butchered from P2PNet to get this running quickly)

import numpy as np
import xmltodict
import re
import json
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
import os
import glob
from PIL import Image
import time
import math
import scipy.io as scio
import argparse
import shutil
import misc.format_data as format_data
# from tqdm import tqdm


def get_points(root_path, mat_path):
    # m = scio.loadmat(os.path.join(root_path, mat_path)) # broken path
    m = scio.loadmat(mat_path) 
    return m['image_info'], m['labels']


def get_image_list(root_path, sub_path):
    images_path = os.path.join(root_path, sub_path, 'images')
    # images = [os.path.join(images_path, im) for im in os.listdir(os.path.join(root_path, images_path)) if 'png' in im] # broken, concats root path twice
    images = [os.path.join(images_path, im) for im in os.listdir(images_path) if 'png' in im] 
    return images


def get_gt_from_image(image_path):
    gt_path = os.path.dirname(image_path.replace('images', 'ground-truth'))
    gt_filename = os.path.basename(image_path)
    gt_filename = 'GT_{}'.format(gt_filename.replace('png', 'mat'))
    return os.path.join(gt_path, gt_filename)


def convert_data_for_P2PNet(root_path, exp_name):
    output_path = root_path + "processed_P2PNET_data/" + exp_name + "/"
    root_path = root_path + "processed_data/" + exp_name + "/"

    dataset_splits = ['train', 'test']
    for split in dataset_splits:
        sub_path = '{}_data'.format(split)
        images = get_image_list(root_path, sub_path=sub_path)

        try:
            os.makedirs(os.path.join(output_path, sub_path))

        except FileExistsError:
            print('Warning, output path already exists, overwriting')

        list_file = []
        for i, image_path in enumerate(images):
            print('\tProcessing %d/%d: ' % (i, len(images)))
            gt_path = get_gt_from_image(image_path)

            if not os.path.exists(gt_path):
                continue

            gt, labs = get_points(root_path, gt_path)  # returns ground truths and person labels

            # for each image, generate a txt file with annotations
            new_labels_file = os.path.join(output_path, sub_path, os.path.basename(image_path).replace('png', 'txt'))
            with open(new_labels_file, 'w') as fp:
                for p, l in zip(gt, labs):
                    fp.write('{} {} {}\n'.format(p[0], p[1], int(l[0])))

            list_file.append((image_path, new_labels_file))

        # generate file with listing
        with open(os.path.join(output_path, '{}.list'.format(split)), 'w') as fp:
            for item in list_file:
                fp.write('{} {}\n'.format(item[0], item[1]))


def convert_data_for_DKPNet(root_dir, exp_name, is_train, model):
    dataset = 'SLSQ'
    train_test_for_gt_SH = '/train_data' if is_train else '/test_data'
    train_test_for_den = '/den/train' if is_train else '/den/test'

    image_dir_path = root_dir + "processed_data/" + exp_name + train_test_for_gt_SH + "/images"
    ground_truth_dir_path = root_dir + "processed_data/" + exp_name + train_test_for_gt_SH + "/ground-truth"

    output_den_path = root_dir + "processed_%s_data/" % model + exp_name + train_test_for_den
    output_img_path = root_dir + "processed_%s_data/" % model + exp_name + "/ori" + train_test_for_gt_SH + "/images"
    output_mat_path = root_dir + 'processed_%s_data/' % model + exp_name + "/ori" + train_test_for_gt_SH + "/ground_truth"

    mkdirs(output_den_path)
    mkdirs(output_img_path)
    mkdirs(output_mat_path)

    img_paths = None
    img_paths = glob.glob(image_dir_path + "/*")


    print("#--------------------------------------------------#")
    print('-------Extracting Annotations for %s --------' % model)
    print("#--------------------------------------------------#")

    for i, img_path in enumerate(img_paths):
        if dataset == 'SLSQ':
            gt_path = os.path.join(ground_truth_dir_path, "GT_" + os.path.basename(img_path)[:-4] + ".mat")

        else:
            assert 1 == 2

        print('\tProcessing %d/%d: ' % (i, len(img_paths)))

        img = Image.open(img_path).convert('RGB')
        height = img.size[1]
        width = img.size[0]

        if os.path.exists(gt_path):

            if dataset == 'SLSQ':
                points = scio.loadmat(gt_path)['image_info']
            else:
                assert 1 == 2

        else:
            continue

        resize_height = height
        resize_width = width

        if dataset == 'SLSQ':
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width

            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32

        else:
            assert 1 == 2

        ratio_h = (resize_height) / (height)
        ratio_w = (resize_width) / (width)
        gt = get_density_map_gaussian(resize_height, resize_width, ratio_h, ratio_w, points, 4)
        gt = np.reshape(gt, [resize_height, resize_width])  # transpose into w, h

        # transfer gt to float16 to save storage
        gt = gt.astype(np.float16)

        # Three stuffs to store
        # 1. images with new folders
        os.system('cp ' + img_path + ' ' + os.path.join(output_img_path, dataset + '_' + os.path.basename(img_path)))
        # 2. save density maps
        np.save(os.path.join(output_den_path, dataset + "_" + os.path.basename(img_path)[:-4] + ".npy"),
                gt)  # some extensions are '.JPG', so...
        # 3. save mats
        scio.savemat(os.path.join(output_mat_path, dataset + "_" + os.path.basename(img_path)[:-4] + ".mat"),
                     {'annPoints': points})

    print("complete!")

def get_density_map_gaussian(H, W, ratio_h, ratio_w,  points, fixed_value=15):
    h = H
    w = W
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    #print('points shape', points.shape)

    for idx, p in enumerate(points):

        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
        sigma = fixed_value
        sigma = max(1, sigma)

        gaussian_radius = 7
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def read_xml(xml_file_path, save_dir, im_path, max_len, min_len, seq_num):

    # skip_files = ['29_01_2022_9_56_17_00006.png', '29_01_2022_9_56_17_00005.png']

    with open(xml_file_path) as fd:
        doc = xmltodict.parse(fd.read())

    images = doc['annotations']['image']

    prev_img_num = 99999999
    img_list = [] # add a list that keeps track of consecutive images, that determines if a sequence is long enough before adding it
    # for im in tqdm(range(len(images)), desc=f"Processing seq {seq_num}"):
    for im in range(len(images)):
        image = images[im]
        image_name = image['@name']
        img_num = int(image_name[-9:-4]) # get ID of image

        print('\t processing %d/%d - img ID %d' % (im+1, len(images), img_num)) # debugging

        # if image_name in skip_files:
        #     continue

        if "points" in image.keys():
            # Loop over images to find valid sequences, discarding images that do not meet min_len
            if (img_num == (prev_img_num + 1) and len(img_list) < max_len) or not img_list: 
                img_list.append(image)
            else:
                if len(img_list) >= min_len:
                    process_sequence(img_list, save_dir, im_path, seq_num)
                    seq_num += 1
                else: # debugging
                    print("\t Not enough images, skipping sequence")

                img_list.clear()
                img_list.append(image)

            prev_img_num = img_num 

        else: # CURRENTLY, NEGATIVE SAMPLES (no people in image) ARE NOT UTILISED FOR TRAINING
            # print('\t processing %d/%d - seq %d' % (im+1, len(images), seq_num))
            print('\t -- No points, skipping')
            continue

    return (seq_num + 1) # go to next sequence for next beach
    

def process_sequence(img_list, save_dir, im_path, seq_num):
    """Process a valid sequence and store images/annotations in the desired location.

    Params:
    img_list: The list of images in the sequence
    save_dir: The path to the directory to save the sequences
    im_path: The path to the images to be processed.
    seq_num: Number of current sequence being saved.
    """    
   
    label_dict = ['none', 'head', 'head_swimming', 'head_boardrider']
    
    print('\t Saving seq %d - len %d images' % (seq_num, len(img_list))) # debugging

    for image in img_list:
        image_name = image['@name']
        img = cv2.imread(os.path.join(im_path, image_name))

        # if seq_counter % (max_len) == 0 and im != 0: # divide into sequences
        #     seq_num += 1
        #     seq_counter = 0

        # # check if sequence is long enough when at end of folder (probably redundant now)
        # if im == (len(images) - 1) and seq_counter == 0:
        #     print('Only 1 img for seq, skipping')
        #     return (seq_num)
        
        # seq_counter += 1

        points = image['points']

        point_list = []
        labels_list = []

        if type(points) != type([]):  # 'collections.OrderedDict':
            head_label = points['@label']
            head_coord = points['@points']  # str
            labels_list.append(head_label)  # person type ('none', 'head', 'head_swimming', 'head_boardrider')
            point_list.append(head_coord)

        else:
            for p in range(len(points)):
                point = points[p]
                head_label = point['@label']
                head_coord = point['@points']
                labels_list.append(head_label)
                point_list.append(head_coord)

        regions = []

        points_arr = np.zeros((len(point_list), 2))
        labels_arr = np.zeros((len(labels_list), 1))

        i = 0
        for p in point_list:
            x, y = p.split(',')
            x, y = float(x), float(y)

            points_arr[i, 0] = x
            points_arr[i, 1] = y

            i += 1

            # -- image plot --
            # plt.scatter(x, y,  s=4)
            img = cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

            ret = {}
            ret['name'] = "rect"
            ret['x'] = x
            ret['y'] = y
            ret['width'] = 100
            ret['height'] = 100

            bbox = {}
            bbox["shape_attributes"] = ret
            bbox["region_attributes"] = {}

            regions.append(bbox)

        i = 0
        for lab in labels_list:

            lab_id = int(label_dict.index(lab))
            labels_arr[i] = lab_id

            i += 1

        # mat_path = os.path.join(save_dir + 'ground-truth/'+ 'GT_' + image_name.replace('png', 'mat')) # old path
        # scio.savemat(npy_path, {'image_info': points_arr, 'labels': labels_arr})

        # Save labels and images in sequence
        seq_gt_path = os.path.join(save_dir, 'ground-truth/', str(seq_num))            
        mkdir(seq_gt_path)
        npy_path = os.path.join(seq_gt_path, image_name.replace('png', 'npy')) # with sequence IDs
        np.save(npy_path, np.hstack((points_arr,labels_arr)))

        img = cv2.imread(os.path.join(im_path, image_name))
        seq_im_path = os.path.join(save_dir, 'images/', str(seq_num))            
        mkdir(seq_im_path)
        cv2.imwrite(os.path.join(seq_im_path, image_name), img)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='../dataset/SLSQ/',
                        help="Root path for the dataset")
    parser.add_argument('--dataset', default='SLSQ', type=str)
    parser.add_argument("--exp_name", default='large_person_type', help="Name of the experiment. ")
    # parser.add_argument("--model", default='DKPNET', help="Model that we are extracting features for. ")
    parser.add_argument('--max_len', default=5, type=int, help="Maximum sequence length. ") # default is 4
    parser.add_argument('--min_len', default=2, type=int, help="Minimum sequence length. ") 
    parser.add_argument("--HPC", default=False, action="store_true", help="Whether to unpack data to HPC local storage first. ") 
    args = parser.parse_args()

    if args.exp_name == 'surfersparadise' or args.exp_name == 'surfersparadise_person_type':
        train_set = ['surfersparadise_05_02_2022']
        test_set = ['surfersparadise_05_02_2022_b']

    elif args.exp_name == 'noosa' or args.exp_name == 'noosa_person_type':
        train_set = ['noosa_08_02_2022']
        test_set = ['noosa_08_02_2022_b']

    elif args.exp_name == 'large' or args.exp_name == 'large_person_type':
        train_set = ['surfersparadise_05_02_2022', 'noosa_08_02_2022', 'portdouglas_6_02_2022', 'portdouglas_08_02_2022', 'sunshinebeach_29_01_2022', 'sunshinebeach_02_02_2022']
        test_set = ['surfersparadise_05_02_2022_b', 'surfers_paradise_sunrise1' ,'noosa_08_02_2022_b', 'noosa_sunset_1', 'noosa_sunset_2', 'portdouglas_08_02_2022_b', 'mooloolaba_sunrise', 'bribie_island_sunrise', 'sunshinebeach_29_01_2022_b']

    elif args.exp_name == 'LOSO_site1':  # Leave surfers out
        train_set = ['noosa_08_02_2022', 'portdouglas_6_02_2022', 'portdouglas_08_02_2022', 'noosa_08_02_2022_b', 'portdouglas_08_02_2022_b', 'sunshinebeach_29_01_2022_b', 'sunshinebeach_29_01_2022', 'sunshinebeach_02_02_2022', 'noosa_sunset_1', 'noosa_sunset_2']
        test_set = ['surfersparadise_05_02_2022', 'surfersparadise_05_02_2022_b', 'surfers_paradise_sunrise1']
    elif args.exp_name == 'LOSO_site2':  # Leave sunshine beach out
        train_set = ['noosa_08_02_2022', 'portdouglas_6_02_2022', 'portdouglas_08_02_2022', 'noosa_08_02_2022_b', 'portdouglas_08_02_2022_b', 'surfersparadise_05_02_2022', 'surfersparadise_05_02_2022_b', 'surfers_paradise_sunrise1', 'noosa_sunset_1', 'noosa_sunset_2']
        test_set = ['sunshinebeach_29_01_2022_b', 'sunshinebeach_29_01_2022', 'sunshinebeach_02_02_2022']
    elif args.exp_name == 'LOSO_site3':  # Leave portdouglas  out
        train_set = ['sunshinebeach_29_01_2022_b', 'sunshinebeach_29_01_2022', 'sunshinebeach_02_02_2022', 'noosa_08_02_2022',  'noosa_08_02_2022_b', 'surfersparadise_05_02_2022', 'surfersparadise_05_02_2022_b', 'surfers_paradise_sunrise1', 'noosa_sunset_1', 'noosa_sunset_2']
        test_set = ['portdouglas_6_02_2022', 'portdouglas_08_02_2022', 'portdouglas_08_02_2022_b']
    elif args.exp_name == 'LOSO_site4':  # Leave noosa out
        train_set = ['portdouglas_6_02_2022', 'portdouglas_08_02_2022', 'portdouglas_08_02_2022_b', 'sunshinebeach_29_01_2022_b', 'sunshinebeach_29_01_2022', 'sunshinebeach_02_02_2022', 'surfersparadise_05_02_2022', 'surfersparadise_05_02_2022_b', 'surfers_paradise_sunrise1']
        test_set = ['noosa_08_02_2022',  'noosa_08_02_2022_b', 'noosa_sunset_1', 'noosa_sunset_2']

    if args.HPC:
        args.root_dir = '/data1/STGN-SLSQ'
        format_data.unzip_data(args.root_dir, args.dataset) # unzip data to local HPC storage
        args.root_dir = os.path.join(args.root_dir, args.dataset)

    try:  
        train_data_save_dir = os.path.join(args.root_dir, 'processed_data/', args.exp_name, 'train_data/')
        test_data_save_dir = os.path.join(args.root_dir, 'processed_data/', args.exp_name, 'test_data/')

        # Delete existing directories, start with clean slate
        if os.path.exists(train_data_save_dir):
            shutil.rmtree(train_data_save_dir)
        if os.path.exists(test_data_save_dir):
            shutil.rmtree(test_data_save_dir)

        mkdir(os.path.join(train_data_save_dir, 'ground-truth/'))
        mkdir(os.path.join(train_data_save_dir, 'images/'))
        mkdir(os.path.join(test_data_save_dir, 'ground-truth/'))
        mkdir(os.path.join(test_data_save_dir, 'images/'))

        # if not os.path.exists(train_data_save_dir + 'ground-truth/'):
        #     os.makedirs(train_data_save_dir + 'ground-truth/')

        # if not os.path.exists(train_data_save_dir + 'images/'):
        #     os.makedirs(train_data_save_dir + 'images/')

        # if not os.path.exists(test_data_save_dir + 'ground-truth/'):
        #     os.makedirs(test_data_save_dir + 'ground-truth/')

        # if not os.path.exists(test_data_save_dir + 'images/'):
        #     os.makedirs(test_data_save_dir + 'images/')

        # extracting the annotations from xml files
        print('extracting annotations of training set ....')
        seq_num = 0 # set the initial count for each sequence, tracked between folders
        for folder in train_set:

            print("#--------------------------------------------------#")
            print('-----------processing %s----------' % folder)
            print("#--------------------------------------------------#")

            annot_file_path = os.path.join(args.root_dir, 'annots/{}.xml'.format(folder))
            img_path = os.path.join(args.root_dir, 'images', folder)
            # img_path = os.path.join(args.root_dir, 'images/', folder)

            seq_num = read_xml(annot_file_path, train_data_save_dir, img_path, args.max_len, args.min_len, seq_num)

        print('extracting annotations of testing set ....')
        seq_num = 0 # reset sequence count
        for folder in test_set:

            print("#--------------------------------------------------#")
            print('-----------processing %s----------' % folder)
            print("#--------------------------------------------------#")

            annot_file_path = os.path.join(args.root_dir, 'annots/{}.xml'.format(folder))
            img_path = os.path.join(args.root_dir, 'images', folder)

            seq_num = read_xml(annot_file_path, test_data_save_dir, img_path, args.max_len, args.min_len, seq_num)
    
    except Exception as exc:
        if args.HPC:
            format_data.cleanup(args.root_dir)
        raise RuntimeError(f"Error during processing data: {exc}") from exc
            

    # # converting the extracted annotations to DKPNET format
    # print('processing extracted annotations of training set of DKPNET....')
    # convert_data_for_DKPNet(args.root_dir, args.exp_name, 1, args.model)
    # print('processing extracted annotations of testing  set of DKPNET....')
    # convert_data_for_DKPNet(args.root_dir, args.exp_name, 0, args.model)
    # # converting the extracted annotations to P2PNET format
    # print('processing extracted annotations for P2PNET....')
    # convert_data_for_P2PNet(args.root_dir, args.exp_name)


