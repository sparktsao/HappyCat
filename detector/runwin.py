# date: 2016/04/01
# username: spark
# description: running
import os
import sys
import pefile
import imp
from keras.models import model_from_json
import numpy as np
import timeit

def transform_to_numpy_arr(arr):
    nf = len(arr)
    ncol = 1024
    if nf:
        ncol = len(arr[0].split(","))
    data = np.empty((nf,ncol),dtype="float32")
    for i,ar in enumerate(arr):
        data[i,:] = np.asarray(ar.split(","),dtype="float32")
    return data

def sha1OfFile(filepath):
    import hashlib
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

if len(sys.argv)>=5:
    fmodel = sys.argv[1]
    fw = sys.argv[2]
    scanroot = sys.argv[3]
    feature_module = sys.argv[4]
elif len(sys.argv)==2:
    fmodel = "init_model.json"
    fw = "init_nnw.hdf5"
    feature_module = "F_IMT005_ex"
    scanroot = sys.argv[1]
else:
    print "Complex mode"
    print "Usage:"+sys.argv[0]+" modulename weightfile folder featuremodel" 
    print "Default mode"
    print "Usage:"+sys.argv[0]+" folder"
    exit()

FRESULT = sys.argv[0]+".detection_log"
fwlog = open(FRESULT,"a")
# load classifier
if fmodel.endswith(".json"):
    model = model_from_json(open(fmodel).read())
    model.load_weights(fw)
elif fmodel.endswith(".pkl"):
    from sklearn.externals import joblib
    model = joblib.load(fmodel)
else:
    print "Unknown model extention, only accept *.json(deepleanring) *.pkl(sklearn)"
    exit()

fp, pathname, description = imp.find_module(feature_module)
example_package = imp.load_module(feature_module, fp, pathname, description)
feature_buffer_function = example_package.GetFeatureFromBuffer
feature_file_function = example_package.GetFeatureFromFile

# build scan list
Scanlist = []
def searchfiles(bpath):
    try:
        for filename in os.listdir(bpath):
            filenamewp = bpath+os.path.sep+filename
            if os.path.isdir(filenamewp):
                searchfiles(filenamewp)
            else:            
                if filename.endswith(".exe") | filename.endswith(".dll") | (len(filename)==64):
                    hnd = open(filenamewp,"rb")
                    if hnd.read(2) == "MZ":
                        Scanlist.append(filenamewp)        
                    hnd.close()
    except Exception, e:
        fwlog.write("#\t"+bpath+"\t"+str(e)+"\n")
searchfiles(scanroot)

print Scanlist

scanid = 0
# scan
for f1 in Scanlist:
    t0 = timeit.default_timer()
    a1 = feature_file_function(f1)
    t1 = timeit.default_timer()
    predicted = model.predict(transform_to_numpy_arr([a1]))
    print predicted,a1
    t2 = timeit.default_timer()    
    tparse = t1-t0
    tpredi = t2-t1
    fwlog.write(f1+"\t"+sha1OfFile(f1)+"\t"+str(predicted[0][1])+"\t"+str(tparse)+"\t"+str(tpredi)+"\n")
    scanid=scanid+1
    if scanid%10==0: fwlog.flush()
fwlog.close()    
 
