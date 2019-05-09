#coding:utf-8
import os
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

from keras_tqdm import TQDMNotebookCallback

from settings import *
import eb_models
import preprocessing

batch_size = 128
num_classes = 1
epochs = 150
cut_len = 968

def create_train_dataset():
    data0_list = []
    data1_list = []
    for sector in [1, 2, 3, 5]:
        npzpath0 = os.path.join(datdir, "%s_0.npz" % sector)
        npzpath1 = os.path.join(datdir, "%s_1.npz" % sector)
        data1 = load_npz(npzpath1)
        data0 = load_npz(npzpath0)[:data1.shape[0]]
        data0 = make_dataset(data0, cut_len)
        data1 = make_dataset(data1, cut_len)
        data0_list.append(data0)
        data1_list.append(data1)
    data0 = np.vstack(tuple(data0_list))
    data1 = np.vstack(tuple(data1_list))
    label0 = np.zeros(data0.shape[0])
    label1 = np.ones(data1.shape[0])
    data = np.vstack((data0, data1))
    label = np.hstack((label0, label1))
    data = normalize(data)
    return data, label

def main():
    #0データ
    data0_list = []
    for sector in [1, 2, 3, 5]:
        npzpath = os.path.join(datdir, "%s_0.npz" % sector)
        data = preprocessing.load_npz(npzpath)
        data = preprocessing.cut_dataset(data, cut_len)
        data0_list.append(data)
    data0 = np.vstack(tuple(data0_list))
    #1データ
    data1_list = []
    for sector in [1, 2, 3, 5]:
        npzpath = os.path.join(datdir, "%s_1.npz" % sector)
        data = preprocessing.load_npz(npzpath)
        data_rev = preprocessing.reverse_data(data)
        data = preprocessing.over_sampling(data, cut_len, 5)
        data_rev = preprocessing.over_sampling(data_rev, cut_len, 5)
        data1_list.extend([data, data_rev])
    data1 = np.vstack(tuple(data1_list))
    label0 = np.zeros(data0.shape[0])
    label1 = np.ones(data1.shape[0])
    train_data = np.vstack((data0, data1))
    train_label = np.hstack((label0, label1))
    # x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # y_train = np.reshape(y_train, (y_train.shape[0], 1))
    # y_test = np.reshape(y_test, (y_test.shape[0], 1))
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    train_label = np.reshape(train_label, (train_label.shape[0], 1))
    model = eb_models.Model().model
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # h5path = os.path.join(modeldir, "model.h5")
    npzpath = os.path.join(modeldir, "model.npz")
    model.save(h5path, include_optimizer=True)
    np.savez(npzpath, loss=history.history["loss"], val_loss=history.history["val_loss"])

def try_network_model():
    data, label = create_train_dataset()
    #分割
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    #keras用にデータをreshape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    x_train.shape, x_test.shape, y_train.shape, y_test.shape
    params = {
        "conv1" : [8, 16],
        "conv2" : [16, 32],
        "conv3" : [32, 64],
        "kernel" : [2, 5],
    }
    for conv1, conv2, conv3, kernel in product([8, 16], [16, 32], [32, 64], [2, 5]):
        models = [eb_models.Model1(conv1, conv2, conv3, kernel).model,
                  eb_models.Model2(conv1, conv2, conv3, 64, kernel).model,
                  eb_models.Model3(conv1, conv2, conv3, kernel).model,
                  eb_models.Model4(conv1, conv2, conv3, 64, kernel).model,
                  ]
        for i, model in enumerate(models):
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
            h5path = os.path.join(modeldir, "%s_%s_%s_%s_%s.h5" % (conv1, conv2, conv3, kernel, i))
            npzpath = os.path.join(modeldir, "%s_%s_%s_%s_%s.npz" % (conv1, conv2, conv3, kernel, i))
            model.save(h5path, include_optimizer=True)
            np.savez(npzpath, loss=history.history["loss"], val_loss=history.history["val_loss"])

if __name__ == '__main__':
    main()
