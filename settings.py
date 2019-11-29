#coding:utf-8
import os

# tessroot = "/manta/pipeline"
tessroot = "/stingray/pipeline"
# tessroot = "/media/tomoyuki-tajiri/tempdir"
# datroot = "C:\\Users\\tajiri\\Desktop\\tess\\data"
# datroot = "/home/tomoyuki-tajiri/univ/tess/data"
datroot = "/home/tajiri/tess/data"
manta = "/manta/pipeline"
stingray = "/stingray/pipeline"

rootdir = os.path.abspath(os.path.dirname(__file__))
CTLdir = os.path.join(tessroot, "CTL2")
TICdir = os.path.join(tessroot, "TIC2")

datdir = os.path.join(datroot, "dat")
modeldir = os.path.join(datroot, "model")

csvdir = os.path.join(datroot, "csv")
allcsvdir = os.path.join(csvdir, "all")
predcsvdir = os.path.join(csvdir, "pred")
VIcsvdir = os.path.join(csvdir, "VI")
traincsvdir = os.path.join(csvdir, "train")
