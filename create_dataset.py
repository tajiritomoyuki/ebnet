#coding:utf-8
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import h5py

from settings import *

# csvname = "CTL4.csv"

def load_lc(path):
    """
    hdf5ファイルのpathを指定してlcデータを返す
    """
    h5path1 = os.path.join(CTLdir, path)
    h5path2 = os.path.join(TICdir, path)
    if os.path.exists(h5path1):
        h5path = h5path1
    else:
        h5path = h5path2
    with h5py.File(h5path, "r") as f:
        flux = np.array(f["LC"]["SAP_FLUX"])
        quality = np.array(f["TPF"]["QUALITY"])
        mid_val = np.nanmedian(flux)
        if mid_val == 0:
            lc = np.zeros_like(flux)
        else:
            lc = flux / mid_val
        lc_interp = np.copy(lc)
        x = lambda z: z.nonzero()[0]
        lc_interp[quality] = np.interp(x(quality), x(~quality), lc_interp[~quality])
    return lc_interp

def create_train(csvpath):
    df = pd.read_csv(csvpath)
    f_sector = lambda x: int(x.split("_")[2])
    sector_df = df["path"].map(f_sector)
    df["sector"] = sector_df
    sectors = sector_df.unique()
    labels = df["label"].unique()
    for label, sector in product(labels, sectors):
        tar_df = df[(df["sector"] == sector) & (df["label"] == label)]
        path_list = tar_df["path"].values
        lc_list = []
        #読み込み
        with mp.Pool(mp.cpu_count()) as p:
            for lc in p.imap(load_lc, tqdm(path_list)):
                lc_list.append(lc)
        lc_array = np.vstack(tuple(lc_list))
        #書き出し
        dstpath = os.path.join(datdir, "%s_%s.npz" % (sector, label))
        np.savez(dstpath, data=lc_array)

def create_test(csvpath):
    df = pd.read_csv(csvpath)
    f_sector = lambda x: int(x.split("_")[2])
    sector_df = df["path"].map(f_sector)
    df["sector"] = sector_df
    path_list = df["path"].values
    lc_list = []
    #読み込み
    with mp.Pool(mp.cpu_count()) as p:
        for lc in p.imap(load_lc, tqdm(path_list)):
            lc_list.append(lc)
    lc_array = np.vstack(tuple(lc_list))
    #書き出し
    csvname = os.path.splitext(os.path.basename(csvpath))[0]
    dstpath = os.path.join(datdir, "%s.npz" % csvname)
    np.savez(dstpath, data=lc_array, path=path_list)

if __name__ == '__main__':
    csvlist = ["CTL11.csv", "CTL12.csv"]
    for csvname in csvlist:
        csvpath = os.path.join(allcsvdir, csvname)
        create_test(csvpath)
