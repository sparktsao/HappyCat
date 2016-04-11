# date: 2016/04/01
# username: spark
# description: feature extrator
import pefile
import sys
import os
import hashlib

ConfigWorkDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
FEATURE_MAX = 1024
FEATURE_FILE = "/F_IMT001.txt"

def T_Buffer2Namearray(buffer1):                                    
    try:                                                  
        p1 = pefile.PE(data=buffer1)                  
    except:                                               
        return ["CannotOpenByPENone:None"]                
    if not hasattr(p1,'DIRECTORY_ENTRY_IMPORT'):          
        return ["CanopenButNone:None"]              
    iarr = []                                             
    for imp in p1.DIRECTORY_ENTRY_IMPORT:
        for t1 in imp.imports:
            if t1.name==None:
                continue
            if t1.name.strip()>0:
                iarr.append(imp.dll.strip()+":"+t1.name.strip())
    return iarr

def CREATE_DICTIONARY_FROM_PRESET(selected):
    mydict = {}
    id = 0
    for name in selected:
        mydict[name] = id
        id = id+1
    return mydict        

def IMT_HASH(str):
    v = hashlib.md5(str).hexdigest()
    rv = 0
    rv = int(v[:(1 + (FEATURE_MAX / 256))*2], 16) % FEATURE_MAX    
    return rv

def T_NameArray2IndexArray(NArr):
    result = []
    base = len(PRESET_DICT)
    rest = FEATURE_MAX - base
    for name in Narr:
        name = name.strip()
        if len(name)==0:
            continue
        hashv = 0
        if name in PRESET_DICT.keys():
            hashv = PRESET_DICT[name]
        else:
            hashv = IMT_HASH(name)%rest+base
        result.append(hashv)
    return result

HASHDIC = {}

def addHASHDIC(hashv,name):
    if HASHDIC.has_key(hashv):
        nameset = HASHDIC[hashv]
        nameset.add(name)        
    else:
        nameset = set([name])
        HASHDIC[hashv]= nameset        
        
def T_NameArray2FeatureVector(NArr):
    Farr = FEATURE_MAX*[0]
    Farr1 = [0]
    base = len(PRESET_DICT)
    rest = FEATURE_MAX - base
    for name in NArr:
        if name =='CannotOpenByPENone:None':
            Farr1[0] = 1
        elif name =='CanopenButNone:None':
            Farr1[0] = 0.5
        else:
            tmp = name.split(":")
            funname = name
            if len(tmp)>1:
                funname = name.split(":")[1]
            hashv = 0
            if funname in PRESET_DICT.keys():
                hashv = PRESET_DICT[funname]
            else:
                hashv = IMT_HASH(funname)%rest+base
            addHASHDIC(  hashv,name)
            Farr[hashv]=1
    return ','.join([str(x) for x in Farr+Farr1])

def T_CleanFunctionName(line):
    return line.strip().replace("[","").replace("'","").replace(",","").replace("]","")
                
def READ_PRESET_DATA_FROMFILE():
    global FEATURE_FILE
    try:
        FEATURE_FILE = ConfigWorkDir + FEATURE_FILE
        f1 = open(FEATURE_FILE)
        lines = f1.readlines()        
        #print "#\tFEATURE FILE %s Used" % FEATURE_FILE
        f1.close()
    except:     
        return {}
    selected = []
    for line in lines:
        name = T_CleanFunctionName(line)
        if len(name)>0:
            selected.append(name)
    PRESET_DICT = CREATE_DICTIONARY_FROM_PRESET(selected)
    return PRESET_DICT

PRESET_DICT = READ_PRESET_DATA_FROMFILE()

def T_IndexArray2FeatureVector(sparsearr):
    result = [0]*FEATURE_MAX
    for item in sparsearr:
        result[item] = 1
    return ",".join(result)
                
def GetFeatureFromBuffer(buffer1):
    arr = T_Buffer2Namearray(buffer1)
    return T_NameArray2FeatureVector(arr)

def GetFeatureFromFile(fname):
    f1 = open(fname,"rb")
    b1 = f1.read()
    f1.close()
    return GetFeatureFromBuffer(b1)

def GetMetaFromFile(fname):
    f1 = open(fname,"rb")
    b1 = f1.read()
    f1.close()
    arr = T_Buffer2Namearray(b1)
    return ",".join(arr)

def GetFeatureFromLine(line):
    arr = line.split(",")
    return T_NameArray2FeatureVector(arr)

"""
File -> buffer -> NameArray -> IndexArray -> FeatureVector
"""

def PARSE_SHA1_RAW_FILE_BYFOLDER(fstring):
    # PARSE SHA1-RAW
    flist = os.listdir(fstring)
    for sha1 in flist:
        if os.path.isfile(fstring+sha1):
            if len(sha1)==40:
                f1 = open(fstring+sha1)
                functions = f1.readlines()
                dataarr = [x.strip() for x in functions]
                print sha1+","+T_NameArray2FeatureVector(dataarr)

def PARSE_SINGLE_FLOG(fname):
    f1 = open(fname)
    while 1:
        line = f1.readline()
        if not line: break
        if line[0] == "#": continue
        z1 = line.strip().split(",")
        sha1 = z1[0]
        dataarr = z1[1:]
        print sha1+","+T_NameArray2FeatureVector(dataarr)
    f1.close()

if __name__ == '__main__':

    if len(sys.argv)==1:
        print "Usage:"+sys.argv[0]+" FLOG/PE/META FOLDER/FILE"
        exit()

    ACTION = sys.argv[1]
    TARGET = sys.argv[2]
    if ACTION=='FLOG':
        if not os.path.isdir(TARGET):
            PARSE_SINGLE_FLOG(TARGET)
        else:
            PARSE_SHA1_RAW_FILE_BYFOLDER(TARGET)
    elif ACTION=='PE':
        if not os.path.isdir(TARGET):
            print GetFeatureFromFile(TARGET)
        else:
            flist = os.listdir(TARGET)
            for f in flist:
                if os.path.isfile(TARGET+f):
                    print f+","+GetFeatureFromFile(TARGET+f)        
    elif ACTION =='META':
        if not os.path.isdir(TARGET):
            print GetMetaFromFile(TARGET)
        else:
            flist = os.listdir(TARGET)
            for f in flist:
                if os.path.isfile(TARGET+f):
                    print f+","+GetMetaFromFile(TARGET+f)
    bGenerateTable = True
    if bGenerateTable:        
        f1 = open("__FunctionHashTable.table","a")
        for key in HASHDIC:
            nameset = list(HASHDIC[key])
            f1.write( str(key)+ ","+",".join(nameset) +"\n" )
        f1.close()

