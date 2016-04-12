# date: 2016/04/01
# username: spark
# description: learning fundation

import os
import sys

import numpy as np
import timeit

from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

from sklearn.cross_validation import KFold

import pandas

import learning_kernel as sparkcore

#========================================
# General parameters
sparkcore.bBalance = False
dropcolumns = []
#========================================

if len(sys.argv)>=2:
    path1 = sys.argv[1]
else:
    print "Please run as %s vlog-folder" % sys.argv[0]
    exit(1)
if len(sys.argv)>=3:
    sparkcore.nb_folds = int(sys.argv[2])
if len(sys.argv)>=4:
    sparkcore.NOTE = sys.argv[3]
if len(sys.argv)>=5:
    dropcolumns= sys.argv[4].split(",")

class RandomCat:
    def fit(self,trainingdata,traininglabel):
        return 0
    def predict(self,testdata):
        import random
        r = [random.randint(0,1) for x in testdata]
        return np.asarray(r)
    def predict_proba(self,testdata):
        return self.predict(testdata)
    def to_json(self):
        return "meow"
    def save_weights(self,a,overwrite=True):
        return "meow"
       
def TrainAndValidation1(X_train,y_train,X_test,y_test,bEarlyStopByTestData=True):
    
    print "Training shape:" , X_train.shape
    print "Training label:" , y_train.shape   

    model = RandomCat() 
    model.fit(X_train,y_train.ravel())
    if not X_test is None:
        predicted = model.predict(X_test)
        v_precision,v_recall,TP, FP, TN, FN,thelogloss = sparkcore.MyEvaluation(y_test,predicted)
        return 0,0,v_precision,v_recall,TP, FP, TN, FN,thelogloss, model
    else:
        return 0,0,0,0,0,0,0,0,0, model


if  __name__ == '__main__':    
    bMulticlass = False
    logdata = sparkcore.ExpFunc(path1,TrainAndValidation1)
    #============================================================================================    
    logdata.logModel = ("")
    #============================================================================================
    logdata.doprint()



