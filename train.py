import argparse, random, time, os, pdb
from datetime import datetime
import numpy as np
from numpy.random import shuffle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torch.nn.functional as F

import np_transforms as NP_T
from CrowdDataset import CrowdSeq
from model import STGN

import shutil


def main():
    parser = argparse.ArgumentParser(
        description='Train CSRNet in Crowd dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='STGN-SLSQ', type=str)
    parser.add_argument('--dataset', default='SLSQ', type=str)
    parser.add_argument('--valid', default=0, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--max_len', default=5, type=int, help="Max sequence length to train on. ") # default is 4
    parser.add_argument('--min_len', default=2, type=int, help="Minimum sequence length to train on. ")
    # parser.add_argument('--max_len', default=None, type=int) # Can leave as None, process_data.py ensures sequence length
    parser.add_argument('--channel', default=128, type=int)
    parser.add_argument('--block_num', default=4, type=int)
    parser.add_argument('--shape', default=[360, 640], nargs='+', type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_all', action='store_true', help='')
    parser.add_argument('--adaptive', action='store_true', help='')
    parser.add_argument('--agg', action='store_true', help='')
    parser.add_argument('--use_cuda', default=True, type=bool)
    # parser.add_argument("--exp_name", default='large_person_type', help="Name of the experiment. ")
    parser.add_argument("--exp_name", default='noosa', help="Name of the experiment. ")
    parser.add_argument("--HPC", default=False, action="store_true", help="Whether to unpack data to HPC local storage first. ") 

    args = vars(parser.parse_args())
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
    
    save_path = os.path.join('models', args['dataset'], args['exp_name'])
    print(args)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_time = datetime.now()
    save_string = f"{args['model_name']}_ep{args['epochs']}_min{args['min_len']}max{args['max_len']}_{save_time.strftime('%Y-%m-%d_%H-%M')}.pth"
    log_path = os.path.join(save_path, save_string + ".txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, 'w') as f:
        f.write(str(args) + '\n')

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    # if args['use_cuda'] == True and we have a GPU, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda']
                          and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    train_transf = NP_T.ToTensor()
    valid_transf = NP_T.ToTensor()

    # instantiate the dataset
    # dataset_path = os.path.join('E://code//Traffic//Counting//Datasets', args['dataset']) # original path

    # change root if using HPC local storage
    if args['HPC']:
        dataset_root = '/data1/STGN-SLSQ/'
    else:
        dataset_root = '../dataset'

    if args['dataset'] == 'SLSQ':
        dataset_path = os.path.join(dataset_root, args['dataset'], 'processed_data', args['exp_name'])
    else:
        dataset_path = os.path.join(dataset_root, args['dataset'])

    train_data = CrowdSeq(train=True,
                          path=dataset_path,
                          out_shape=args['shape'],
                          transform=train_transf,
                          gamma=args['gamma'],
                          max_len=args['max_len'],
                          load_all=args['load_all'],
                          adaptive=args['adaptive'])
    valid_data = CrowdSeq(train=False,
                          path=dataset_path,
                          out_shape=args['shape'],
                          transform=valid_transf,
                          gamma=args['gamma'],
                          max_len=args['max_len'],
                          load_all=args['load_all'],
                         adaptive=args['adaptive'])


    train_loader = DataLoader(train_data,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=6)

    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=6)
    
    model = STGN(args).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args['epochs'])

    # training routine
    t_start = time.time() 
    for epoch in range(args['epochs']):
        print_epoch = 'Epoch {}/{}'.format(epoch + 1, args['epochs'])
        print(print_epoch, flush=True)
        with open(log_path, 'a') as f:
            f.write(print_epoch + '\n')

        # training phase
        model.train()
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        X, density, count = None, None, None
        t0 = time.time()
        for i, (X, density, count, seq_len) in enumerate(train_loader):
            X, density, count, seq_len = X.to(device), density.to(
                device), count.to(device), seq_len.to(device)
            
            b, t, c, h, w = X.shape

            if random.random() < 0.5:
                X = torch.flip(X, [-1])
                density = torch.flip(density, [-1])

            density_pred, count_pred = model(X)
            N = torch.sum(seq_len)
            count = count.sum(dim=[2,3,4])
            density_loss = torch.sum((density_pred - density)**2) / (2 * N)
            count_loss = torch.sum((count_pred - count)**2) / (2 * N)
            loss = density_loss

            # backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            with torch.no_grad():
                count_err = torch.sum(torch.abs(count_pred - count)) / N # MAE
            count_err_hist.append(count_err.item())
        lr_scheduler.step()
        t1 = time.time()

        train_loss = sum(loss_hist) / len(loss_hist)
        train_density_loss = sum(density_loss_hist) / len(density_loss_hist)
        train_count_loss = sum(count_loss_hist) / len(count_loss_hist)
        train_count_err = sum(count_err_hist) / len(count_err_hist)
        print('Training statistics:', flush=True)
        log = '{} - density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_density_loss, train_count_loss, train_count_err)
        print(log, flush=True)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

        # validation phase
        model.eval()
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        mse_hist = []
        mae_hist = []
        X, density, count = None, None, None
        t0 = time.time()
        val_loss = 0
        for i, (X, density, count, seq_len) in enumerate(valid_loader):
            X, density, count, seq_len = X.to(device), density.to(
                device), count.to(device), seq_len.to(device)

            # forward pass through the model
            with torch.no_grad():
                density_pred, count_pred = model(X)

            # compute the loss
            N = torch.sum(seq_len)
            count = count.sum(dim=[2,3,4])
            count_loss = torch.sum((count_pred - count)**2) / (2 * N) # basically MSE
            
            density_loss = torch.sum((density_pred - density)**2) / (2 * N) # basically MSE
            loss = density_loss

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            mae = torch.sum(torch.abs(count_pred - count)) / N
            mae_hist.append(mae.item())
            mse = torch.sum((count_pred - count)**2) / N
            mse_hist.append(mse.item())
        t1 = time.time()

        # print the average validation losses
        valid_loss = sum(loss_hist) / len(loss_hist)
        valid_density_loss = sum(density_loss_hist) / len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist) / len(count_loss_hist)
        valid_mse = sum(mse_hist) / len(mse_hist)
        valid_rmse = np.sqrt(valid_mse)
        valid_mae = sum(mae_hist) / len(mae_hist)
        if epoch == 0:
            min_mae = valid_mae
            min_mse = valid_mse
            min_rmse = valid_rmse
            min_epoch = epoch + 1
        else:
            if valid_mae <= min_mae:
                min_mae = valid_mae
                min_mse = valid_mse
                min_rmse = valid_rmse
                min_epoch = epoch + 1
                best_path = os.path.join(save_path, save_string)
                torch.save(
                    model.state_dict(),
                    best_path)

        print('Validation statistics:', flush=True)
        log = '{} - density loss: {:.3f} | count loss: {:.3f} | MAE: {:.3f} | MSE: {:.3f} | RMSE: {:.3f}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), valid_density_loss, valid_count_loss, valid_mae, valid_mse, valid_rmse)
        print(log, flush=True)
        with open(log_path, 'a') as f:
            f.write(log + '\n')

    t_end = time.time()

    # Save final model in generic path
    log = 'Final Results:\nBest MAE: {:.3f} | Best MSE: {:.3f} | Best RMSE: {:.3f} | Epoch # {} | Training time: {:.2f} seconds'.format(min_mae, min_mse, min_rmse, min_epoch, t_end-t_start)
    print(log, flush=True)
    with open(log_path, 'a') as f:
            f.write(log + '\n')

    model_generic_path = os.path.join('models', args['dataset'], 'STGN_latest.pth')
    model_final_path = os.path.join(save_path, save_string)
    shutil.copyfile(model_final_path, model_generic_path)


if __name__ == '__main__':
    main()
