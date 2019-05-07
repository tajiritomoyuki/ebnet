#coding:utf-8
import os
import csv
import glob

from settings import *

def main():
    sector = 4
    h5list = glob.glob(os.path.join(CTLdir, "*_%s_?_?.h5" % sector))
    csvpath = os.path.join(csvdir, "CTL%s.csv" % sector)
    with open(csvpath, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path"])
        for h5path in h5list:
            writer.writerow([os.path.basename(h5path)])

if __name__ == '__main__':
    main()