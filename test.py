import argparse, random, time, os, pdb
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

import np_transforms as NP_T
from CrowdDataset import TestSeq
from model import STGN
from sklearn.metrics import mean_squared_error,mean_absolute_error

def main():
    parser = argparse.ArgumentParser(
        description='Train CSRNet in Crowd dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--model_path', default="STGN_2022-10-01.pth", type=str, help="Model file to be tested. Do not include path to the model.")
    parser.add_argument('--model_path', default=None, type=str, help="Model file to be tested. Leave as `None` to test most recent model")
    parser.add_argument('--dataset', default='SLSQ', type=str)
    parser.add_argument('--valid', default=0, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=5, type=int) # default is 4
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--shape', default=[360, 640], nargs='+', type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_all', action='store_true', help='')
    parser.add_argument('--adaptive', action='store_true', help='')
    parser.add_argument('--agg', action='store_true', help='')
    parser.add_argument('--use_cuda', default=True, type=bool)
    # parser.add_argument("--exp_name", default='large_person_type', help="Name of the experiment. ")
    parser.add_argument("--exp_name", default='noosa', help="Name of the experiment.")
    parser.add_argument("--HPC", default=False, action="store_true", help="Whether to unpack data to HPC local storage first. ") 

    args = vars(parser.parse_args())
    
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    valid_transf = NP_T.ToTensor() 

# ONLY TESTING SLSQ DATASET, NOT OTHERS
    # datasets = ['TRANCOS', 'Venice', 'UCSD', 'Mall', 'FDST'] 
    # for dataset in datasets:
    #     if dataset == 'UCSD':
    #         args['shape'] = [360, 480]
    #         args['max_len'] = 10
    #         args['channel'] = 128
    #     elif dataset == 'Mall':
    #         args['shape'] = [480, 640]
    #         args['max_len'] = 4
    #         args['channel'] = 128
    #     elif dataset == 'FDST':
    #         args['max_len'] = 4
    #         args['shape'] = [360, 640]
    #         args['channel'] = 128
    #     elif dataset == 'Venice':
    #         args['max_len'] = 8
    #         args['shape'] = [360, 640]
    #         args['channel'] = 128
    #     elif dataset == 'TRANCOS':
    #         args['max_len'] = 4
    #         args['shape'] = [360, 480]
    #         args['channel'] = 128

    if args['dataset'] == 'UCSD':
        args['shape'] = [360, 480]
    elif args['dataset'] == 'Mall':
        args['shape'] = [480, 640]
    elif args['dataset'] == 'FDST':
        args['shape'] = [360, 640]
    elif args['dataset'] == 'Venice':
        args['shape'] = [360, 640]
    elif args['dataset'] == 'TRANCOS':
        args['shape'] = [360, 480]
    # elif args['dataset'] == 'SLSQ':
    #     args['shape'] = [720, 1280] # possibly increase this to [720, 1280] if GPU can handle it
            
    # dataset_path = os.path.join('E://code//Traffic//Counting//Datasets', dataset)

    # change root if using HPC local storage
    if args['HPC']:
        dataset_root = '/data1/STGN-SLSQ/'
    else:
        dataset_root = '../dataset'

    if args['dataset'] == 'SLSQ':
        dataset_path = os.path.join(dataset_root, args['dataset'], 'processed_data', args['exp_name'])
    else:
        dataset_path = os.path.join(dataset_root, args['dataset'])

    valid_data = TestSeq(train=False,
                            path=dataset_path,
                            out_shape=args['shape'],
                            transform=valid_transf,
                            gamma=args['gamma'],
                            max_len=args['max_len'], 
                            load_all=args['load_all'])
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1)

    model = STGN(args).to(device)
    model.eval()
    # get model to test
    if args['model_path'] is None:
        model_full_path = os.path.join('./models', args['dataset'], 'STGN_latest.pth')
    else:
        model_full_path = os.path.join('./models', args['dataset'], args['exp_name'], args['model_path'])
    assert os.path.exists(model_full_path) is True
    model.load_state_dict(torch.load(model_full_path))
    # print('Load pre-trained model')

    X, density, count = None, None, None
    
    preds = {}
    predictions = []
    counts = []
    t1 = time.time()
    for i, (X, count, seq_len, names) in enumerate(valid_loader):
        X, count, seq_len = X.to(device), count.to(device), seq_len.to(device)

        with torch.no_grad():
            density_pred, count_pred = model(X)
    
        N = torch.sum(seq_len)
        count = count.sum(dim=[2,3,4])
        count_pred = count_pred.data.cpu().numpy()
        count = count.data.cpu().numpy()

        for i, name in enumerate(names):
            dir_name, img_name = name[0].split('&')
            preds[dir_name + '_' + img_name] = count[0, i]
            
            predictions.append(count_pred[0, i])
            counts.append(count[0, i])
    t2 = time.time()
    test_time = t2-t1

    mae = mean_absolute_error(predictions, counts)
    mse = mean_squared_error(predictions, counts)
    rmse = np.sqrt(mse)
    
    print('Dataset: {} | MAE: {:.3f} | MSE: {:.3f} | RMSE: {:.3f} | Eval time: {:.2f} seconds'.format(args['dataset'], mae, mse, rmse, test_time))

        
if __name__ == '__main__':
    main()
