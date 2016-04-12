# date: 2016/04/01
# username: spark tsao
# reviewer: ace wu
# description: learning fundation
import sys
import os
import timeit
import datetime
import pandas
import random
import logging

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import KFold

FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

KLABEL = {"malicious":1,"normal":0}
WORKING_KLABEL = 2
KLABEL_FOCUS = "malicious"
B_MULTICLASS = False
RANDOM_STATE = 23310838
#========================================
# General parameters
nb_folds = 10
bBalance = False
NOTE = 'youcanreplaceme'

OUTPUT_DLOG_NAME='__DEEPWRS_learning.elog'
OUTPUT_SHA1_SCORE_NAME=''#'expect_result.csv'
OUTPUT_MODEL_NAME=''
OUTPUT_MODEL_W_NAME=''

ARR_score0 = []
ARR_score1 = []
ARR_rtime = []
ARR_tr_precision = []
ARR_tr_recall = []
ARR_4N = []
APR_logloss = []
#========================================

class Transformer:

    def __init__(self,name,ia,ib,func):

        self.name = name
        self.ia = ia
        self.ib = ib
        self.func = func

    def cut_n_apply(self,data):
        c1 = data.iloc[:,self.ia:self.ib]
        if hasattr(self.func, '__call__'):
            c = c1.apply(self.func,axis=1)
            c1 = c
        return c1

def split_transform_combine(data,transformers):

    parts = []
    for t in transformers:
        a = t.cut_n_apply(data)
        parts.append(  a  )
    newparts = pandas.concat(parts,axis=1)
    return newparts

def parsefile(f,flag,transformer):
    logging.info( "#\tPandas Parsefile:%s\t%s", f, flag)
    f1 = pandas.read_csv(f, comment="#", header=None, engine='c', dtype={0: 'object'})
    f1.columns = ["sha1"]+f1.columns.tolist()[1:]
    f1 = f1[f1.sha1!='sha1'] # remove possible header
    sha1list = f1.sha1.tolist()
    data = f1.ix[:,1:] # remove sha1
    data = data.fillna(0)

    # transform
    if len(transformer):
        data = split_transform_combine(data,transformer)

    da = np.array(data,dtype="float32")
    return da,sha1list

def PrepareLabel1(n, is_malicious):
    if is_malicious:
        return np.ones(n, dtype=np.int)
    else:
        return np.zeros(n, dtype=np.int)

def PrepareLabelN(n, glabel, ng):
    label = np.zeros((n, ng), dtype=np.int)
    label[:, glabel] = 1
    return label

def is_collection(x):
    try:
        i = iter(x)
    except TypeError:
        return False
    else:
        return not isinstance(x, (basestring, bytes))

def Process(path1,bBalance=False,bIsMulticlass=True,dropcolumns=[],transformers=[]):
    ngroup = 0
    f_list = os.listdir(path1)
    if f_list != None:
        f_list.sort()
    for fname in f_list:
        if not fname.endswith(".vlog"): continue
        flabel = fname.split(".")[0]
        if flabel not in KLABEL: continue

        ngroup = ngroup + 1

    global WORKING_KLABEL
    WORKING_KLABEL = ngroup

    data = None
    label = None
    sha1list = None
    for fname in f_list:
        if not fname.endswith(".vlog"): continue
        flabel = fname.split(".")[0]
        if flabel not in KLABEL: continue
        ix = KLABEL[flabel]
        if not bIsMulticlass and ix>1: continue
        d2, s1 = parsefile(os.path.join(path1, fname), ix,transformers)
        if bIsMulticlass:
            l1 = PrepareLabelN(d2.shape[0],ix,ngroup)
        else:
            l1 = PrepareLabel1(d2.shape[0],ix)
        l2 = l1 #np.array(l1,dtype="uint8")
        s2 = np.array(s1,dtype="object")
        if data is None:
            data = d2
            label = l2
            sha1list = s2
        else:
            data = np.append(data,d2,axis=0)
            label = np.append(label,l2,axis=0)
            sha1list = np.append(sha1list,s2,axis=0)

    if data is None or len(data) == 0:
        logging.error("no data")
        exit()

    random.seed(1571)
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    sha1list = sha1list[index]
    
    if len(dropcolumns)>0:
        s1 =  data.shape
        dropcolumns = [int(x)-1 for x in dropcolumns] # shift x-1 due to remove sha1
        data = np.delete(data,dropcolumns,axis=1)
        logging.info( "#\tdrop %s, %s to %s", str(s1), str(dropcolumns),str(data.shape))

    return data,label,sha1list
        
def perf_measure(y_actual, y_hat,focusix):
    TP,FP,TN,FN = 0,0,0,0
    if is_collection(y_actual[0])==False:
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

import scipy as sp
def logloss_1(act, pred):
    act = act.flatten()
    pred = pred.flatten()
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
def myargmax(x):
    ret = [0.0]*len(x)
    import numpy
    id = numpy.argmax(x)
    ret[id] = 1.0
    return ret


def logloss(act,pred):
    #import numpy
    #pred1 = numpy.asarray( [myargmax(x) for x in pred] )
    ##print pred1, pred, act
    #r1 = logloss_1(act,pred1)
    r2 = logloss_1(act,pred)
    #print r1,r2
    return r2





def MyEvaluation(y_test,predicted):
    mylogloss = logloss(y_test,predicted)

    def norm_me(x):
        if 'int' in str(type(x)):
            return x
        if not is_collection(x):
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
    logging.info(result)

    averagelabel = 'binary'
    if B_MULTICLASS: averagelabel = 'macro'
    (TP, FP, TN, FN) = perf_measure(y_test, predicted,KLABEL[KLABEL_FOCUS])

    v_precision = 0 if (TP+FP)==0 else TP*1.0/(TP+FP)
    v_recall = 0 if (TP+FN)==0  else  TP*1.0/(TP+FN)
    logging.info("#\tKLABEL:"+str( KLABEL)+ str((v_precision,v_recall,TP, FP, TN, FN)) +" logloss:"+str(mylogloss))

    return v_precision,v_recall,TP, FP, TN, FN, mylogloss

def getSaveNames(learner,datapath,note,score):

    if len(OUTPUT_MODEL_NAME) and len(OUTPUT_MODEL_W_NAME)>0:
        return OUTPUT_MODEL_NAME,OUTPUT_MODEL_W_NAME

    m_name = learner.split('.', 1)[0]
    fname = datapath.replace("/","-").replace(".","").replace("\\\\","\\").replace("\\","-").strip("-")
    fnameMODEL = "M="+ m_name+"="+fname+"="+note

    maxprecision = min(4,len(str(score)))
    fnameWeight = fnameMODEL+'=E'+str(score)[0:maxprecision]

    return fnameMODEL,fnameWeight

def Dump(model,fnameMODEL,fnameWeight):
    if 'keras.model' in str(type(model)):
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD
        json_string = model.to_json()
        fm = open(fnameMODEL+".json","w")
        fm.write(json_string)
        fm.close()

        model.save_weights(fnameWeight+".hdf5",overwrite=True)
    elif 'sklearn.' in str(type(model)):
        from sklearn.externals import joblib
        def ensure_dir(f):
            d = os.path.dirname(f)
            if not os.path.exists(d):
                os.makedirs(d)
        #ensure_dir('./skmodel/')
        joblib.dump(model, fnameMODEL+".pkl",compress=1)
    else:
        logging.warning("unkown model type:"+str(type(model))+" found, your should implement your DumpFunction")

def Log(message):
    with open(OUTPUT_DLOG_NAME, 'a') as fout:
        print >>fout, '\t'.join([str(x) for x in message])

def release():
    global OUTPUT_SHA1_SCORE_NAME,  OUTPUT_MODEL_NAME, OUTPUT_MODEL_W_NAME
    OUTPUT_SHA1_SCORE_NAME='release_result.csv'
    OUTPUT_MODEL_NAME='model'
    OUTPUT_MODEL_W_NAME='modelweight'

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
        self.loglogloss = ()
        self.logBVT = ()# BestValidationTest
        self.logTrainingTime = ()# ('Time',timearr,np.mean(timearr) )
        self.logModel =  ()# ('DEEP',DEEP_AVF,DEEP_DROPOUTR,DEEP_SGDLR,DEEP_EPOCH,DEEP_BSIZE,D_CLASS_NORMALW)
        self.logNote = ()#  ('NOTE',NOTE)
        self.logSave = ()# ('Model',fnameMODEL,fnameWeight)
        self.logTest = ()
        self.logTransformerID = ()
        self.logDropcolumns = ()
        self.logRandom = ()

    def getLogList(self):
        logs = [attr for attr in dir(LogData()) if not callable(attr) and not attr.startswith("__")]
        logging.info(logs)

    def doprint(self):
        self.logModel = ("MODEL",":".join( [str(x) for x in self.logModel] ) )
        self.logSave = ("SAVE",":".join( [str(x) for x in self.logSave] ) )

        Log(self.logCMD + self.logData + self.logEXP + self.logLoad + self.logTrainingErr +
                self.logValidationErr + self.logPrecision + self.logRecall + self.logR4N + self.loglogloss +
                self.logTrainingTime + self.logModel + self.logNote + self.logSave + self.logTest + self.logBVT + self.logTransformerID + self.logDropcolumns + self.logRandom)

def CalcF1Score(p,r):
    if (p+r)==0: return 0
    return 2.0*p*r/(p+r)

def DoKFold(data,label,myLearnandValidate):
    model = None
    f1max = 0
    kfolds = KFold( data.shape[0] , nb_folds)
    for FID, (trainix, validix) in enumerate(kfolds):
        logging.info( "#\tFold:%s total %s", str(FID), str(nb_folds) )

        t2 = timeit.default_timer()
        X_train = data[trainix]
        y_train = label[trainix]
        X_test = data[validix]
        y_test = label[validix]

        score0,score1,v_precision,v_recall,TP, FP, TN, FN, mylogloss, model = myLearnandValidate(X_train,y_train,X_test,y_test)

        ARR_score0.append(score0)
        ARR_score1.append(score1)
        ARR_4N.append( ",".join([str(x) for x in (TP, FP, TN, FN)] ) )
        ARR_tr_precision.append(v_precision)
        ARR_tr_recall.append(v_recall)
        APR_logloss.append(mylogloss)

        t3 = timeit.default_timer()
        ARR_rtime.append(t3-t2)

        f1 = CalcF1Score(v_precision,v_recall)
        if f1>f1max:
            f1max = f1
            model = model
        logging.info( "#\t %s %s %s %s %s %s %s %s",str(f1),str(v_precision),str(v_recall),str(TP), str(FP), str(TN), str(FN),str(mylogloss))

    return f1,model

def ExpFunc(path1,myLearnandValidate,bIsMulticlass=False,shaper=None):

    global B_MULTICLASS
    B_MULTICLASS = bIsMulticlass

    transformer = []
    dropcolumns = []
    transformerid = ''
    dropstring = ''
    for item in sys.argv:
        if item.startswith("transformer="):
            transformerid = item[12:]
            try:
                sys.path.insert(0, '../featurex/')
                import transform_manager
                transformer = transform_manager.getTransformers(transformerid)
            except:
                logging.warning("transformer not found:%s",item)
        elif item.startswith("dropcolumns="):
            try:
                dropstring =  item[12:]
                dropcolumns = dropstring.split(",") 
            except:
                logging.warning("dropcolumns parameters error:%s",item)
        elif item =='release':
            release()
        elif item.startswith("randomstate="):
            global RANDOM_STATE
            try:
                rs = int( item.split("=")[1] )
                RANDOM_STATE = rs
            except:
                rs = 0


    # LOAD DATA
    t0 = timeit.default_timer()
    data, label, sha1list= Process(path1,bBalance,bIsMulticlass,dropcolumns,transformer)
    if not shaper is None:
        data = shaper(data)

    t1 = timeit.default_timer()

    # DO K FOLD
    f1 = 0
    model_bv = None
    if nb_folds>0:
        f1, model_bv = DoKFold(data,label,myLearnandValidate)

    # LOAD Test Data
    if os.path.exists(os.path.join(path1, 'test')):
        tdata, tlabel, tsha1 = Process(os.path.join(path1, 'test'), bBalance, bIsMulticlass, dropcolumns, transformer)
        if not shaper is None:
            tdata = shaper(tdata)
    else:
        tdata, tlabel, tsha1 = None, None, None

    # Test using the best fold
    f3,vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN = 0,0,0,0,0,0,0
    if not tdata is None and not model_bv is None:    
        predicted_vb = model_bv.predict(tdata)
        vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN, vblogloss = MyEvaluation(tlabel,predicted_vb)
        f3 = CalcF1Score(vb_precision,vb_recall)

    # TRAINING and TESTING
    score0,score1,t_precision,t_recall,tTP, tFP, tTN, tFN,tlogloss, model = myLearnandValidate(data,label,tdata,tlabel,True)
    f4 = CalcF1Score(t_precision,t_recall)
    logging.info("#\tF1Score: BestFold: %s\tAllData: %s",str(f3),str(f4))

    # Log and Serialization
    fnameMODEL,fnameWeight = getSaveNames(sys.argv[0],sys.argv[1],NOTE,score1)
    Dump(model,fnameMODEL,fnameWeight)

    # Final[Sha1/score]
    if not tdata is None and not model is None:
        predicted_test = model.predict_proba(tdata)
        global OUTPUT_SHA1_SCORE_NAME
        if OUTPUT_SHA1_SCORE_NAME=="": OUTPUT_SHA1_SCORE_NAME = fnameMODEL+".sha1c"
        fpredict = open(OUTPUT_SHA1_SCORE_NAME,"w")
        for (sha1,p) in zip(tsha1,predicted_test):
            fpredict.write(sha1+","+str(p)+"\n")
        fpredict.close()

    tfinal = timeit.default_timer()
    logging.info("#\tLearning success! prepare to return logs :take %s",str(tfinal-t1))

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
    logdata.loglogloss = ("LOGLOSS",APR_logloss)
    logdata.logTrainingTime =  ('TTime',ARR_rtime,np.mean(ARR_rtime) )
    logdata.logNote =   ('NOTE',NOTE)
    logdata.logTest = ("Test",t_precision,t_recall,tTP, tFP, tTN, tFN,tlogloss)
    logdata.logSave = (fnameMODEL,fnameWeight)
    logdata.logBVT = ("BVT",vb_precision,vb_recall,vbTP, vbFP, vbTN, vbFN, vblogloss)
    logdata.logTransformerID = ("TFM",transformerid)
    logdata.logDropcolumns = ("DP",dropstring)
    logdata.logRandom = ("RS",RANDOM_STATE)
    return logdata



