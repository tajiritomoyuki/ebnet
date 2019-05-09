#coding:utf-8
import os
import csv
import glob

from settings import *

def main():
    for sector in [1, 2, 3, 5]:
        # sector = 4
        h5list = glob.glob(os.path.join(CTLdir, "*_%s_?_?.h5" % sector))
        csvpath = os.path.join(csvdir, "CTL%s.csv" % sector)
        with open(csvpath, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["path"])
            for h5path in h5list:
                writer.writerow([os.path.basename(h5path)])

if __name__ == '__main__':
    main()
