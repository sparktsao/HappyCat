#---------------------------
# Spark_Tsao@trend.com.tw
#---------------------------

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
# Deep learning parameters
DEEP_AVF = 'relu'
DEEP_DROPOUTR = 0.5
DEEP_SGDLR = 0.01
DEEP_EPOCH = 20
DEEP_BSIZE = 16
D_CLASS_NORMALW = 1
DEEP_LOSSFUNC = "mean_squared_error"
DEEP_EARLY_STOP_P = 10
DEEP_VALIDATION_SPLIT = 0.01
DEEP_MaxEpochInTraining = 0
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

def TrainAndValidation1(X_train,y_train,X_test,y_test,bEarlyStopByTestData=True):    
    
    print "#\tTraining shape:" , X_train.shape
    print "#\tTraining label:" , y_train.shape   
    #============================================
    # Model preparation
    #============================================
    model = Sequential()
    model.add(Dense(output_dim=64,input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation(DEEP_AVF))

    model.add(Dense(64, init='uniform'))
    model.add(Activation(DEEP_AVF))

    model.add(Dense(sparkcore.WORKING_KLABEL, init='uniform'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=DEEP_SGDLR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=DEEP_LOSSFUNC, optimizer=sgd)
    
    a = model.fit(X_train, y_train,nb_epoch=5)
    print a
    score0 = model.evaluate(X_train, y_train, batch_size=DEEP_BSIZE)

    if not X_test is None:
        score1  = model.evaluate(X_test, y_test, batch_size=DEEP_BSIZE)
        predicted = model.predict(X_test)
        v_precision,v_recall,TP, FP, TN, FN,thelogloss = sparkcore.MyEvaluation(y_test,predicted)
        return score0,score1,v_precision,v_recall,TP, FP, TN, FN, thelogloss,model
    else:
        return score0, 0,0,0,0,0,0,0,0,model        

if  __name__ == '__main__':    
    
    logdata = sparkcore.ExpFunc(path1,TrainAndValidation1,True)
    #============================================================================================    
    logdata.logModel = (("Activation",DEEP_AVF),("Dropout",DEEP_DROPOUTR),("lr",DEEP_SGDLR),("nb_epoch",DEEP_EPOCH),("",DEEP_MaxEpochInTraining),("batch_size",DEEP_BSIZE),("class_weight",D_CLASS_NORMALW),("",DEEP_EARLY_STOP_P))
    #============================================================================================
    logdata.doprint()
