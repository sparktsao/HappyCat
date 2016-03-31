# date: 2016/04/01
# username: spark
# description: learning fundation
import sys
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import KFold
import timeit
import datetime
import pandas
import random

KLABEL = {"malicious":1,"normal":0,"spark":2}
WORKING_KLABEL = 2
#========================================
# General parameters
nb_folds = 10
bBalance = False
NOTE = 'youcanreplaceme'

ARR_score0 = []
ARR_score1 = []
ARR_rtime = []
ARR_tr_precision = []
ARR_tr_recall = []
ARR_4N = []
#========================================


def parsefile(f,flag):
    print "#\tPandas Parsefile:", f, flag
    f1 = pandas.read_csv(f,comment="#",header=None,engine='c')
    f1.columns = ["sha1"]+f1.columns.tolist()[1:]
    f1 = f1[f1.sha1!='sha1'] # remove possible header
    
    data = f1.ix[:,1:] # remove sha1
    data = data.fillna(0)
    da = np.array(data,dtype="float64")
    return da


def PrepareLabel1(n,ismalicious):
    if ismalicious:
        l1 = [[1] for i in range(n)]
    else:
        l1 = [[0] for i in range(n)]
    return l1


def PrepareLabelN(nrecords,glabel,ng):
    r1 = [0]*ng
    r1[glabel]=1
    r2 = [r1 for i in range(nrecords)]
    return r2

def Process(path1,bBalance=False,bIsMulticlass=True,dropcolumns=[]):
    ngroup = 0
    for fname in os.listdir(path1):
        if not fname.endswith(".vlog"): continue
        flabel = fname.split(".")[0]
        if False==( flabel in KLABEL): continue

        ngroup = ngroup + 1

    global WORKING_KLABEL
    WORKING_KLABEL = ngroup

    data = None
    label = None    
    for fname in os.listdir(path1):
        if not fname.endswith(".vlog"): continue
        flabel = fname.split(".")[0]
        if False==( flabel in KLABEL): continue
        ix = KLABEL[flabel]
        if bIsMulticlass==False and ix>1: continue
        d2 = parsefile(path1+fname,ix)
        if bIsMulticlass:
            l1 = PrepareLabelN(d2.shape[0],ix,ngroup)
        else:
            l1 = PrepareLabel1(d2.shape[0],ix)
        l2 = np.array(l1,dtype="uint8")
        if data is None:
            data = d2
            label = l2
        else:
            data = np.append(data,d2,axis=0)
            label = np.append(label,l2,axis=0)

    if len(data)==0: 
        print "no data"
        exit()

    random.seed(1571)
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]

    s1 =  data.shape
    dropcolumns = [int(x)-1 for x in dropcolumns] # shift x-1 due to remove sha1
    data = np.delete(data,dropcolumns,axis=1)
    print "#\tdrop", s1, dropcolumns, " to ",data.shape
    return data,label
        

def perf_measure(y_actual, y_hat,focusix):
    TP,FP,TN,FN = 0,0,0,0
    if len(y_actual[0])==1:
        for i in range(len(y_hat)):
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            elif y_actual[i]==1 and y_actual[i]!=y_hat[i]:
               FN += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            elif y_actual[i]==0 and y_actual[i]!=y_hat[i]:
               FP += 1
        return(TP, FP, TN, FN)

    for i in range(len(y_hat)): 
        if np.all(y_actual[i]==y_hat[i]) and y_hat[i][focusix] ==1:
           TP += 1
        elif y_actual[i][focusix]==1 and np.all(y_actual[i]!=y_hat[i]):
           FN += 1
        elif np.all(y_actual[i]==y_hat[i]) and y_hat[i][focusix] ==0:
           TN += 1
        elif y_actual[i][focusix]==0 and np.all(y_actual[i]!=y_hat[i]):
           FP += 1

    return(TP, FP, TN, FN)

def MyEvaluation(y_test,predicted):
    def norm_me(x):
        if str(type(x)).find("int")>-1:
            return x
        zix = np.argmax(x)
        x1 = [0]*len(x)
        x1[zix] = 1
        return x1
    predicted = [norm_me(x) for x in predicted]
    predicted = np.array(predicted,dtype="uint8")

    target_names  = ['normal','malware']
    inv_map = {v: k for k, v in KLABEL.items()}
    target_names = [inv_map[x] for x in range(WORKING_KLABEL)]
    result = classification_report(y_test,predicted,target_names=target_names)
    print result

    v_precision = precision_score(y_test,predicted, average="macro")#, average='binary')
    v_recall = recall_score(y_test,predicted, average="macro")#, average='binary')    

    (TP, FP, TN, FN) = perf_measure(y_test, predicted,KLABEL["malicious"])
    return v_precision,v_recall,TP, FP, TN, FN

def getSaveNames(learner,datapath,note,score):
    m_name = learner.split(".")[0]
    fname = datapath.replace("/","-").replace(".","").strip("-")
    fnameMODEL = "M_"+ m_name+"_"+fname+"_"+note

    maxprecision = min(4,len(str(score)))
    fnameWeight = fnameMODEL+'_E'+str(score)[0:maxprecision]

    return fnameMODEL,fnameWeight

def Dump(model,fnameMODEL,fnameWeight):
    if str(type(model)).find("sklearn.")==-1:
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD
        json_string = model.to_json()
        fm = open(fnameMODEL+".json","w")
        fm.write(json_string)
        fm.close()
    
        model.save_weights(fnameWeight+".hdf5",overwrite=True)
    else:
        from sklearn.externals import joblib
        def ensure_dir(f):
            d = os.path.dirname(f)
            if not os.path.exists(d):
                os.makedirs(d)
        ensure_dir('./skmodel/')
        joblib.dump(model, "./skmodel/"+fnameMODEL+".pkl",compress=3)

def Log(message):
    fout = open("__DEEPWRS_learning.elog","a")
    fout.write(  "\t".join( [str(x) for x in message]  )+"\n" )
    fout.close()

class LogData:

    def __init__(self):
        self.logCMD = ()#("CMD",sys.argv[0],sys.argv[1])
        self.logData = ()# ('DATA',data.shape)
        self.logEXP = ()# ('EXP',nb_folds)
        self.logLoad = ()# ('PEF',t0,t1-t0)
        self.logTrainingErr = ()# ('T',scorearr0,np.mean(scorearr0),np.std(scorearr0))
        self.logValidationErr = ()# ('V',scorearr,np.mean(scorearr),np.std(scorearr))
        self.logPrecision = ()# ('Precision',precision_arr,np.mean(precision_arr),np.std(precision_arr))
        self.logRecall =  ()# ('Recall',recall_arr,np.mean(recall_arr),np.std(recall_arr))
        self.logR4N =  ()#("T",FourT_arr)
        self.logBVT = ()# BestValidationTest
        self.logTrainingTime = ()# ('Time',timearr,np.mean(timearr) )
        self.logModel =  ()# ('DEEP',DEEP_AVF,DEEP_DROPOUTR,DEEP_SGDLR,DEEP_EPOCH,DEEP_BSIZE,D_CLASS_NORMALW)
        self.logNote = ()#  ('NOTE',NOTE)
        self.logSave = ()# ('Model',fnameMODEL,fnameWeight)
        self.logTest = ()

    def getLogList(self):
        logs = [attr for attr in dir(LogData()) if not callable(attr) and not attr.startswith("__")]
        print logs

    def doprint(self):

        self.logModel = ("MODEL",":".join( [str(x) for x in self.logModel] ) )
        self.logSave = ("SAVE",":".join( [str(x) for x in self.logSave] ) )
        
        Log(self.logCMD+self.logData+self.logEXP+self.logLoad+self.logTrainingErr+self.logValidationErr+self.logPrecision+self.logRecall+self.logR4N+self.logTrainingTime+self.logModel+self.logNote+self.logSave+self.logTest+self.logBVT)        

def CalcF1Score(p,r):
    if (p+r)==0: return 0
    return 2.0*p*r/(p+r)

def DoKFold(data,label,myLearnandValidate):
    model = None
    f1max = 0
    kfolds = KFold( data.shape[0] , nb_folds)
    for FID, (trainix, validix) in enumerate(kfolds):
        print "#\tFold:", FID, " total ", nb_folds

        t2 = timeit.default_timer()
        X_train = data[trainix]
        y_train = label[trainix]
        X_test = data[validix]
        y_test = label[validix]
        
        score0,score1,v_precision,v_recall,TP, FP, TN, FN, model = myLearnandValidate(X_train,y_train,X_test,y_test)

        ARR_score0.append(score0)
        ARR_score1.append(score1)
        ARR_4N.append( ",".join([str(x) for x in (TP, FP, TN, FN)] ) )
        ARR_tr_precision.append(v_precision)
        ARR_tr_recall.append(v_recall)

        t3 = timeit.default_timer()
        ARR_rtime.append(t3-t2)

        f1 = CalcF1Score(v_precision,v_recall) 
        if f1>f1max:
            f1max = f1
            model = model
        print "#\t",f1,v_precision,v_recall,TP, FP, TN, FN

    return f1,model

def ExpFunc(path1,myLearnandValidate,bIsMulticlass=True,dropcolumns=[]):

    # LOAD DATA
    t0 = timeit.default_timer()
    data, label = Process(path1,bBalance,bIsMulticlass,dropcolumns)
    
    t1 = timeit.default_timer()

    # DO K FOLD
    f1 = 0
    model_bv = None
    if nb_folds>0:
        f1, model_bv = DoKFold(data,label,myLearnandValidate)

    # LOAD Test Data
    if os.path.exists(path1+"test"):
        tdata, tlabel = Process(path1+"test/",bBalance,bIsMulticlass,dropcolumns)
    else:
        tdata, tlabel = None, None

    # Test using the best fold
    f3,vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN = 0,0,0,0,0,0,0
    if not tdata is None and not model_bv is None:
        predicted_vb = model_bv.predict(tdata)
        vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN = MyEvaluation(tlabel,predicted_vb)
        f3 = CalcF1Score(vb_precision,vb_recall)

    # TRAINING and TESTING
    score0,score1,t_precision,t_recall,tTP, tFP, tTN, tFN, model = myLearnandValidate(data,label,tdata,tlabel,False)
    f4 = CalcF1Score(t_precision,t_recall)
    print "#\tF1Score: BestFold: ",f3,"\tAllData:",f4

    # Log and Serialization
    fnameMODEL,fnameWeight = getSaveNames(sys.argv[0],sys.argv[1],NOTE,score1)
    Dump(model,fnameMODEL,fnameWeight)

    # LOG
    logdata = LogData()

    logdata.logCMD = ("CMD",sys.argv[0],sys.argv[1])
    logdata.logData = ('DATA',data.shape)
    logdata.logExp = ('EXP',nb_folds)
    tprint = datetime.datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')
    logdata.logLoad = ('LOAD',tprint,t1-t0)

    logdata.logTrainingErr =  ('TrainErr',ARR_score0,np.mean(ARR_score0),np.std(ARR_score0))
    logdata.logValidationErr =  ('ValidErr',ARR_score1,np.mean(ARR_score1),np.std(ARR_score1))
    logdata.logPrecision = ('Precision',ARR_tr_precision,np.mean(ARR_tr_precision),np.std(ARR_tr_precision))
    logdata.logRecall =  ('Recall',ARR_tr_recall,np.mean(ARR_tr_recall),np.std(ARR_tr_recall))
    logdata.logR4N =  ("R4N",ARR_4N)
    logdata.logTrainingTime =  ('TTime',ARR_rtime,np.mean(ARR_rtime) )
    logdata.logNote =   ('NOTE',NOTE)
    logdata.logTest = ("Test",t_precision,t_recall,tTP, tFP, tTN, tFN)
    logdata.logSave = (fnameMODEL,fnameWeight)
    logdata.logBVT = ("BVT",vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN)
    return logdata



