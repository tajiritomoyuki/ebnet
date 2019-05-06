#coding:utf-8
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import h5py

from .settings import *

csvname = ""

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
        quality = np.array(f["LC"]["QUALITY"])
        mid_val = np.nanmedian(flux)
        if mid_val == 0:
            lc = np.zeros_like(flux)
        else:
            lc = flux / mid_val
        lc_interp = np.copy(lc)
        x = lambda z: z.nonzero()[0]
        lc_interp[quality] = np.interp(x(quality), x(~quality), lc_interp[~quality])
    return lc_interp

def main(csvpath):
    df = pd.read_csv(csvpath)
    f_sector = lambda x: int(x.split("_")[2])
    sector_df = df["path"].map(f_sector)
    df["sector"] = sector_df
    #カラム数が2と3の場合で分ける
    #3の場合は訓練データ
    if len(df.columns) == 3:
        sectors = sector_df.unique()
        labels = df["label"].unique()
        for label, sector in product(labels, sectors):
            tar_df = df[(df["sector"] == sector) & (df["label"] == label)]
            path_list = tar_df["path"].values
            lc_list = []
            #読み込み
            with mp.Pool(mp.cpu_count()) as p:
                for lc in p.imap(load, tqdm(path_list)):
                    lc_list.append(lc)
            lc_array = np.vstack(tuple(lc_list))
            #書き出し
            dstpath = os.path.join(datdir, "%s_%s.npz" % (sector, label))
            np.savez(dstpath, data=lc_array)
    #2の場合はテストデータ
    elif len(df.columns) == 2:
        path_list = df["path"].values
        lc_list = []
        #読み込み
        with mp.Pool(mp.cpu_count()) as p:
            for lc in p.imap(load, tqdm(path_list)):
                lc_list.append(lc)
        lc_array = np.vstack(tuple(lc_list))
        #書き出し
        dstpath = os.path.join(datdir, "test.npz")
        np.savez(dstpath, data=lc_array, path=path_list)

if __name__ == '__main__':
    csvpath = os.path.join(csvdir, csvname)
    main(csvpath)
