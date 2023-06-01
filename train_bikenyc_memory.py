# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='4'
import sys
sys.path.append('.')
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import logging
import numpy as np
from datetime import datetime
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../../model')
sys.path.append('../../preprocessing')
from make_dataset import make_dataloader
from utils import weight_init, EarlyStopping, compute_errors
from mstnet1_b import mstnet1
from bikenyc import load_data1_b
import h5py
from torch.utils.data import TensorDataset, DataLoader
#from utils import weight_init, EarlyStopping, compute_errors
import random

#stresnet 纽约数据集

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4 # length of trend dependent sequence
nb_residual_unit = 4  # number of residual units

map_height, map_width = 16, 8  # grid size
nb_flow = 2  # there are two types of flows: new-flow and end-flow
nb_area = 128
len_test=48*7*8

epoch_nums = 500
learning_rate = 0.0002
batch_size = 32
params = {'batch_size': batch_size,
          'shuffle': False,
          'drop_last':False,
          'num_workers':4
          }

test_split=0.1
validation_split = 0.1
early_stop_patience = 100
shuffle_dataset = True

epoch_save = [0 , epoch_nums - 1] \
                + list(range(0, epoch_nums, 50))  # 1*1000


now=datetime.now()
save_path='reports/bikenyc/memory/{date:%Y-%m-%d_%H %M %S}'.format(date=now)
out_dir = './reports/bikenyc'
checkpoint_dir = save_path+'/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter('reports/bikenyc/memory/{date:%Y-%m-%d_%H %M %S}/log'.format(date=now))


LOAD_INITIAL = False
def fix_random_seeds(seed=12):
    torch.manual_seed(seed) #cpu设置
    torch.cuda.manual_seed_all(seed)# gpu设置种子
    np.random.seed(seed)
    random.seed(seed)

def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mae, mse, rmse

def valid(model, val_generator, criterion, device):
    model.eval()
    mae_loss = []
    mse_loss=[]
    rmse_loss=[]
    for i, (X_c, X_p, X_t, X_meta,X_time, Y_batch) in enumerate(val_generator):
        # Move tensors to the configured device
        X_c = X_c.type(torch.FloatTensor).to(device)
        X_p = X_p.type(torch.FloatTensor).to(device)
        X_t = X_t.type(torch.FloatTensor).to(device)
        X_meta = X_meta.type(torch.FloatTensor).to(device)
        #X_time = X_time.type(torch.FloatTensor).to(device)
        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        mae, mse, rmse = criterion(outputs.cpu().data.numpy(), Y_batch.data.numpy())
        mae_loss.append(mae)
        mse_loss.append(mse)
        rmse_loss.append(rmse)

    mae_loss = np.mean(mae_loss)
    mse_loss=np.mean(mse_loss)
    rmse_loss=np.mean(rmse_loss)
    print('Mean test loss:mae {} ,mse {},rmse {}'.format(mae_loss,mse_loss,rmse_loss))
    return mae_loss,mse_loss,rmse_loss

def train():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('training...')
    fix_random_seeds()

    #data loader
    train_dataset = make_dataloader(dataset_name='bikenyc', mode='train',
                                    len_closeness=len_closeness, len_period=len_period,
                                    len_trend=len_trend)
    torch.set_num_threads(3)

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split1 = int(np.floor((1-validation_split-test_split) * dataset_size))
    split2 =int(np.floor((1-test_split) * dataset_size))



    train_indices, val_indices,test_indices = indices[:split1], indices[split1:split2],indices[split2:]
    print('training size:', len(train_indices))
    print('val size:', len(val_indices))



    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    training_generator = data.DataLoader(train_dataset, **params,
                                               sampler=train_sampler)
    val_generator = data.DataLoader(train_dataset, **params,
                                               sampler=valid_sampler)
    test_generator = data.DataLoader(train_dataset, **params,
                                                sampler=test_sampler)

    total_iters=int(len(train_indices)/batch_size)*epoch_nums

    model=mstnet1(learning_rate=learning_rate,
                     c_conf=(len_closeness, nb_flow, map_height, map_width),
                     p_conf=(len_period, nb_flow, map_height, map_width),
                     t_conf=(len_trend, nb_flow , map_height, map_width),
                     external_dim = 8, nb_residual_unit = nb_residual_unit)
    # if LOAD_INITIAL:
    #     logger.info('\tload initial_checkpoint = %s\n' % initial_checkpoint)
    #     model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    #model.apply(weight_init)

    # Loss and optimizer
    loss_fn = nn.MSELoss() # nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    loss_fn.to(device)

    # Train the model
    es = EarlyStopping(patience = early_stop_patience,
                       mode='min', model=model, save_path=checkpoint_dir + '/model.best.pth' )
    for e in range(epoch_nums):
        model.train()
        #adjust_learning_rate=(1-e/epoch_nums)**0.9
        for i, (X_c, X_p, X_t, X_meta,X_time, Y_batch) in enumerate(training_generator):

            # Move tensors to the configured device
            X_c = X_c.type(torch.FloatTensor).to(device)
            X_p = X_p.type(torch.FloatTensor).to(device)
            X_t = X_t.type(torch.FloatTensor).to(device)
            X_meta = X_meta.type(torch.FloatTensor).to(device)
            Y_batch = Y_batch.type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(X_c, X_p, X_t, X_meta)
            loss = loss_fn(outputs, Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        its = np.ceil(len(training_generator) / batch_size) * (e+1)  # iterations at specific epochs
        print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'
              .format(e + 1, epoch_nums, its, total_iters, loss.item()))

        writer.add_scalar('train_loss', loss.item(), e + 1)
        # valid after each training epoch
        mae_loss,mse_loss,rmse_loss = valid(model,val_generator , compute_errors, device)

        writer.add_scalar('test_mae', mae_loss, e + 1)
        writer.add_scalar('test_mse',mse_loss,e+1)
        writer.add_scalar('test_rmse',rmse_loss,e+1)
        print("Mean test  mae(real):%.6f  ,rmse(real):%.6f "%(mae_loss*(train_dataset.mmn._max-train_dataset.mmn._min)/2,rmse_loss*(train_dataset.mmn._max-train_dataset.mmn._min)/2))

        if es.step(mse_loss,e+1):
            print('early stopped! With val loss:', mse_loss)
            print('best_spoch:{}'.format(es.get_best_epoch()))
            break  # early stop criterion is met, we can stop now

        if e in epoch_save:
            torch.save(model.state_dict(), checkpoint_dir + '/%08d_model.pth' % ( e))
            torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': its,
                    'epoch': e,
                }, checkpoint_dir + '/%08d_optimizer.pth' % ( e))

            logger.info(checkpoint_dir + '/%08d_model.pth' % ( e) +
                        ' saved!')

    rmse_list=[]
    mse_list=[]
    mae_list=[]
    y_true=[]
    y_batch=[]
    print(checkpoint_dir)
    print("4层resnet 纽约数据集")
    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir + '/model.best.pth'))

    for i, (X_c, X_p, X_t, X_meta,X_time, Y_batch) in enumerate(val_generator):
        # Move tensors to the configured device
        X_c = X_c.type(torch.FloatTensor).to(device)
        X_p = X_p.type(torch.FloatTensor).to(device)
        X_t = X_t.type(torch.FloatTensor).to(device)
        X_meta = X_meta.type(torch.FloatTensor).to(device)
        y_true.append(Y_batch)
        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        y_batch.append(outputs.cpu().data.numpy())
        mae, mse, rmse = compute_errors(outputs.cpu().data.numpy(), Y_batch.data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    print('Valid mse: %.6f mae: %.6f rmse (norm): %.6f, rmse (real): %.6f ' % (
        mse, mae, rmse, rmse * (train_dataset.mmn._max - train_dataset.mmn._min) / 2.))

    print('Valid mae(real):%.6f'%(mae*(train_dataset.mmn._max-train_dataset.mmn._min)/2.))


    rmse_list = []
    mse_list = []
    mae_list = []
    y_true = []
    y_batch = []
    print(checkpoint_dir)
    print("4层resnet 纽约数据集")
    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir + '/model.best.pth'))

    for i, (X_c, X_p, X_t, X_meta, X_time, Y_batch) in enumerate(test_generator):
        # Move tensors to the configured device
        X_c = X_c.type(torch.FloatTensor).to(device)
        X_p = X_p.type(torch.FloatTensor).to(device)
        X_t = X_t.type(torch.FloatTensor).to(device)
        X_meta = X_meta.type(torch.FloatTensor).to(device)
        y_true.append(Y_batch)
        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        y_batch.append(outputs.cpu().data.numpy())
        mae, mse, rmse = compute_errors(outputs.cpu().data.numpy(), Y_batch.data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    print('Test mse: %.6f mae: %.6f rmse (norm): %.6f, rmse (real): %.6f ' % (
        mse, mae, rmse, rmse * (train_dataset.mmn._max - train_dataset.mmn._min) / 2.))

    print('Test mae(real):%.6f' % (mae * (train_dataset.mmn._max - train_dataset.mmn._min) / 2.))
    writer.close()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()

