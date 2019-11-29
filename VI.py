#-*-coding:utf-8-*-
import os
import glob
import csv
import h5py
import pickle
import MySQLdb
import mysql.connector
from itertools import product
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mpl_toolkits.axes_grid1
from astropy.io import fits
from astropy.coordinates import SkyCoord

from settings import *
from aperture_contour import draw_contours

sql_data = {
      "user" : "fisher",
    "passwd" : "atlantic",
      "host" : "133.11.229.168",
        "db" : "TESS"
}

wcsdir = "/home/tomoyuki-tajiri/univ/tess/ebnet/wcs"
csvdir = "/home/tomoyuki-tajiri/univ/tess/data/csv"
VIdir = "/home/tomoyuki-tajiri/univ/tess/data/csv/VI"
CTLdir = "/media/tomoyuki-tajiri/tempdir/CTL2"
# dstpath = "/home/tomoyuki-tajiri/univ/tess/ebnet/tmp/test.csv"
# h5path1 = "/home/tomoyuki-tajiri/univ/tess/ebnet/tmp/tess_211353706_1_1_1.h5"
# h5path2 = "/home/tomoyuki-tajiri/univ/tess/ebnet/tmp/tess_147085475_1_1_1.h5"

class TinderLight():
    def __init__(self, csvpath):
        self.f = open(csvpath, "a")
        self.writer = csv.writer(self.f, lineterminator="\n")
        self.writer.writerow(["path", "label"])
        #event関連
        self.button_l = False
        self.button_d = False
        self.button_p = False
        self.button_a = False
        self.connect()

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**sql_data)
            self.cursor = self.conn.cursor()
        except:
            self.conn = None
            self.cursor = None

    def check_connection(self):
        if self.conn.is_connected():
            return True
        else:
            self.connect()
            if self.connect is not None:
                return True
            else:
                False

    def set_data(self, h5path):
        self.fn = os.path.basename(h5path)
        with h5py.File(h5path, "r") as ff:
            self.TID = ff["header"]["TID"].value
            self.sector = ff["header"]["sector"].value
            self.camera = ff["header"]["camera"].value
            self.chip = ff["header"]["chip"].value
            self.ra = ff["header"]["ra"].value
            self.dec = ff["header"]["dec"].value
            self.Tmag = ff["header"]["Tmag"].value
            self.x = ff["header"]["x"].value
            self.y = ff["header"]["y"].value
            self.cx = ff["header"]["cx"].value
            self.cy = ff["header"]["cy"].value
            self.time = np.array(ff["LC"]["TIME"])
            sap_flux = np.array(ff["LC"]["SAP_FLUX"])
            self.quality = np.array(ff["LC"]["QUALITY"])
            self.img = np.median(ff["TPF"]["ROW_CNTS"], axis=0)
            self.flux = np.array(ff["TPF"]["FLUX"])
            self.aperture = np.array(ff["APERTURE_MASK"]["FLUX"])
            self.aperture_bkg = np.array(ff["APERTURE_MASK"]["FLUX_BKG"])
            qua1 = np.where(np.mod(self.quality, 2) >= 1, 1 ,0)
            qua4 = np.where(np.mod(self.quality, 8) >= 4, 1 ,0)
            qua = np.logical_or(qua1, qua4)
            self.lc = np.where(qua, np.nan, sap_flux)

    def close_figure(self):
        self.button_l = False
        self.button_d = False
        self.button_p = False
        self.button_q = False
        self.f.flush()
        plt.close()

    def superlike(self, event):
        self.writer.writerow([self.fn, 2])
        self.close_figure()

    def like(self, event):
        self.writer.writerow([self.fn, 1])
        self.close_figure()

    def nope(self, event):
        self.writer.writerow([self.fn, 0])
        self.close_figure()

    def press_event(self, event):
        if event.key == "l":
            self.ax_img.clear()
            self.cax.clear()
            if self.button_l:
                self.button_l = False
                self.draw_img()
            else:
                self.button_l = True
                self.draw_img()
        if event.key == "p" and not self.button_p and self.check_connection():
            self.ax_img.clear()
            self.cax.clear()
            self.button_p = True
            self.draw_img()
        if event.key == "d" and not self.button_d and self.check_connection():
            self.ax_text.clear()
            self.button_d = True
            self.write_header()
        if event.key == "a" and not self.button_d and self.check_connection():
            self.button_d = True
            self.draw_all_lc()
        self.fig.canvas.draw()

    def load_params(self):
        # if not self.conn.is_connected():
        #     self.connect()
        query = "select rad, Teff from CTLchip%s_%s_%s where ID=%s;" % (self.sector, self.camera, self.chip, self.TID)
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result[0]

    def load_position(self):
        # if not self.conn.is_connected():
        #     self.connect()
        wcspath = os.path.join(wcsdir, "%s_%s_%s.pickle" % (self.sector, self.camera, self.chip))
        with open(wcspath, "rb") as f:
            wcs = pickle.load(f)
        ra_max = self.ra + 0.03
        ra_min = self.ra - 0.03
        dec_max = self.dec + 0.03
        dec_min = self.dec - 0.03
        query = "select ra, `dec` from CTLchip%s_%s_%s where ra < %s and ra > %s and `dec` < %s and `dec` > %s;" % (self.sector, self.camera, self.chip, ra_max, ra_min, dec_max, dec_min)
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        xylist = [SkyCoord(ra, dec, unit="deg").to_pixel(wcs) for ra, dec in result]
        positions = [(x - self.x + self.cx, y - self.y + self.cy) for x, y in xylist]
        return positions

    def write_header(self):
        self.ax_text.set_axis_off()
        header = "sector : %s,   camera : %s,   chip : %s,   ra : %s,   dec : %s" % (self.sector, self.camera, self.chip, self.ra, self.dec)
        if self.button_d:
            rad, Teff = self.load_params()
            param = "Tmag : %s,   rad : %s,   Teff : %s" % (self.Tmag, rad, Teff)
        else:
            param = "Tmag : %s" % (self.Tmag)
        self.ax_text.text(0.2, 1., header, size=15)
        self.ax_text.text(0.2, 0.3, param, size=15)

    def draw_img(self):
        if self.button_l:
            img = np.log10(self.img)
        else:
            img = self.img
        im = self.ax_img.imshow(img)
        self.ax_img = draw_contours(self.ax_img, self.aperture, color="white", lw=2)
        self.ax_img = draw_contours(self.ax_img, self.aperture_bkg, color="blue", lw=2)
        self.fig.colorbar(im, ax=self.ax_img, cax=self.cax)
        if self.button_p:
            positions = self.load_position()
            for cx, cy in positions:
                self.ax_img.plot(cx, cy, color="red", marker="o")

    def draw_all_lc(self):
        height, width = self.aperture.shape
        for x, y in product(range(width), range(height)):
            l = self.flux[:, x, y]
            l = np.where(np.mod(self.quality, 2) == 1, np.nan, l)
            if self.aperture[x, y] == 1:
                self.ax_all_lc.plot(self.time, l, color="red")
            else:
                self.ax_all_lc.plot(self.time, l, color="blue")

    def suggest(self):
        self.fig = plt.figure()
        self.fig.suptitle(self.fn, size=15)

        self.ax_text = plt.subplot2grid((10, 4), (0, 0))
        self.ax_all_lc = plt.subplot2grid((10, 4), (1, 0), colspan=3, rowspan=4)
        self.ax_img = plt.subplot2grid((10, 4), (1, 3), rowspan=4)
        self.ax_lc = plt.subplot2grid((10, 4), (5, 0), colspan=4, rowspan=5)

        self.fig.subplots_adjust(hspace=0.8, bottom=0.15)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax_img)
        self.cax = divider.append_axes('right', '5%', pad='3%')

        #プロット
        self.draw_img()
        # self.draw_all_lc()
        self.ax_lc.plot(self.time, self.lc)
        self.write_header()

        axsuperlike = plt.axes([0.65, 0.02, 0.1, 0.075])
        axlike = plt.axes([0.45, 0.02, 0.1, 0.075])
        axnope = plt.axes([0.25, 0.02, 0.1, 0.075])
        button_superlike = Button(axsuperlike, "super like")
        button_superlike.on_clicked(self.superlike)
        button_like = Button(axlike, "like")
        button_like.on_clicked(self.like)
        button_nope = Button(axnope, "nope")
        button_nope.on_clicked(self.nope)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        self.fig.canvas.mpl_connect("key_press_event", self.press_event)
        plt.show()


def test():
    TL = TinderLight(dstpath)
    # h5path = os.path.join(datdir, row[1])
    TL.set_data(h5path1)
    TL.suggest()
    TL.set_data(h5path2)
    TL.suggest()


def main():
    csvpath = os.path.join(predcsvdir, "pred_10.csv")
    dstpath = os.path.join(VIdir, "VI_10.csv")
    with open(csvpath, "r") as f:
        reader = csv.reader(f)
        # for i in range(2735):
        header = next(reader)
        TL = TinderLight(dstpath)
        for i, row in enumerate(reader):
            print(i, row[1], row[2])
            h5path = os.path.join(CTLdir, row[1])
            TL.set_data(h5path)
            TL.suggest()

if __name__ == '__main__':
    main()
