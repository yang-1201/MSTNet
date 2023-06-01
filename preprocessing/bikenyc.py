# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pickle as pickle
import numpy as np

#from . import load_stdata
import os, sys
sys.path.append('.')
sys.path.append('../../preprocessing')
from MaxMinNormalization import MinMaxNormalization
#from preprocessing import remove_incomplete_days
#from .config import Config
from STMatrix import STMatrix
from timestamp import timestamp2vec
import h5py
np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = './data/'
DATAPATH = os.path.dirname(os.path.abspath(__file__))
print(DATAPATH)
DATAPATH = os.path.join(os.path.dirname(DATAPATH), 'data')
CACHEPATH = os.path.join(DATAPATH, 'CACHE_b')
def remove_incomplete_days(data, timestamps, T=24):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps
def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps

def load_data(T=24, nb_flow=2, len_closeness=None, len_period=None, len_trend=None, len_test=None,
              preprocess_name='preprocessing_b.pkl', meta_data=True):
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    data, timestamps = load_stdata(os.path.join(DATAPATH, '', 'NYC14_M16x8_T60_NewEnd.h5'))
    print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    # minmax_scale
    #data_train = data[:-len_test]
    data_train = data
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_closeness=len_closeness, len_period=len_period,
                                                             len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
    X = []

    for l, X_ in zip([len_closeness, len_period, len_trend], [XC, XP, XT]):
        if l > 0:
            X.append(X_)

    # load meta feature
    if meta_data:
        meta_feature ,day_time= timestamp2vec(timestamps_Y)
        day_time = np.asarray(day_time)
        metadata_dim = meta_feature.shape[1]
        X.append(meta_feature)
        X.append(day_time)
    else:
        metadata_dim = None
    print('time feature:', meta_feature.shape, 'meta feature: ', meta_feature.shape)
    print("day feature", day_time.shape)
    print()
    return X, Y,  mmn, metadata_dim, timestamps_Y

def cache(fname, X, Y,  external_dim, timestamp):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X))

    for i, data in enumerate(X):
        h5.create_dataset('X_%i' % i, data=data)

    h5.create_dataset('Y', data=Y)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T', data=timestamp)
    h5.close()



def read_cache(fname,pklname1):
    mmn = pickle.load(open(pklname1, 'rb'))
    f = h5py.File(fname, 'r')
    num = int(f['num'][()])
    X, Y = [], []
    for i in range(num):
        X.append(f['X_%i' % i][()])
    Y = f['Y'][()]
    external_dim = f['external_dim'][()]
    timestamp = f['T'][()]
    f.close()
    return X, Y,  mmn, external_dim, timestamp


def load_data1_b(len_closeness, len_period, len_trend, len_test, meta_data=True, meteorol_data=True, holiday_data=True):
    fname = os.path.join(DATAPATH, CACHEPATH, 'BikeNYC_C{}_P{}_T{}13_17_all1_meta1_t1.h5'.format(len_closeness, len_period, len_trend))
    pklfname1=os.path.join(DATAPATH, CACHEPATH, 'preprocessing_B_{}{}{}_13_17_all1_meta_t1.pkl'.format(len_closeness,len_period,len_trend))
    if os.path.exists(fname):
        X, Y,  mmn, external_dim, timestamp= read_cache(fname,pklfname1)

        print(X[0].shape,X[1].shape,X[2].shape,X[3].shape,X[4].shape,Y.shape,timestamp.shape,mmn._max,mmn._min)

        # (1824, 32, 32, 6)(1824, 32, 32, 2)(1824, 32, 32, 2)(1824, 8)(1824, 32, 32, 2)(1824, )
        # (1344, 32, 32, 6)(1344, 32, 32, 2)(1344, 32, 32, 2)(1344, 8)(1344, 32, 32, 2)(1344, )
        #(3720, 6, 16, 8) (3720, 8, 16, 8) (3720, 8, 16, 8) (3720, 8) (3720,) (3720, 2, 16, 8) (3720,)
        print("load %s successfully" % fname)
    else:
        if os.path.isdir(CACHEPATH) is False:
            os.mkdir(CACHEPATH)
        X, Y,  mmn, external_dim, timestamp= \
            load_data(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                         preprocess_name=pklfname1, meta_data=True)

        cache(fname, X, Y,external_dim, timestamp)
    return X, Y, mmn, external_dim, timestamp


if __name__ == "__main__":

    X, Y,  mmn, external_dim, timestamp = \
        load_data1_b(len_closeness=12, len_period=1, len_trend=1, len_test=28 * 48)

    print(timestamp)
    print(type(X[1]))
    print(mmn._max,mmn._min)
