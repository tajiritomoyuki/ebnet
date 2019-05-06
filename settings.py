#coding:utf-8
import os

tessroot = "/pike/pipeline"
datroot = "C:\\Users\\tajiri\\Desktop\\tess\\data"
datroot = "/home/tajiri/tess/data"

rootdir = os.path.abspath(os.path.dirname(__file__))
CTLdir = os.path.join(tessroot, "step3")
TICdir = os.path.join(tessroot, "TIC3")
csvdir = os.path.join(datroot, "csv")
datdir = os.path.join(datroot, "dat")
modeldir = os.path.join(datroot, "model")
