import pefile
import sys
import os
import hashlib

ConfigWorkDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
FEATURE_MAX = 1024
FEATURE_FILE = 'F_IMT001.txt'
LEARNING_MODE = 0

IMP_CAN_NOT_OPEN_BY_PE_NONE = [('CannotOpenByPENone', 'None')]
IMP_CAN_OPEN_BUT_NONE = [('CanopenButNone', 'None')]

COL_SEP = ','
PAIR_SEP = ':'


def CREATE_DICTIONARY_FROM_PRESET(selected):
    mydict = {}
    id = 0
    for name in selected:
        mydict[name] = id
        id = id + 1
    return mydict


def T_CleanFunctionName(line):
    return line.strip().translate(None, "[]',")


def READ_PRESET_DATA_FROMFILE():
    feature_path = os.path.join(ConfigWorkDir, FEATURE_FILE)
    with open(feature_path) as f1:
        lines = f1.readlines()
    print '#\tFEATURE FILE %s Used' % feature_path

    selected = []
    for line in lines:
        name = T_CleanFunctionName(line)
        if len(name) > 0:
            selected.append(name)
    PRESET_DICT = CREATE_DICTIONARY_FROM_PRESET(selected)
    return PRESET_DICT


PRESET_DICT = READ_PRESET_DATA_FROMFILE()


def T_Buffer2NamePairs(input_buffer):
    try:
        p1 = pefile.PE(data=input_buffer)
    except:
        return IMP_CAN_NOT_OPEN_BY_PE_NONE

    try:
        dir_entry_import = p1.DIRECTORY_ENTRY_IMPORT
    except AttributeError:
        return IMP_CAN_OPEN_BUT_NONE

    pair_list = []
    for imp in dir_entry_import:
        for t1 in imp.imports:
            if t1.name is None:
                continue
            if t1.name.strip() > 0:
                pair_list.append([imp.dll.strip(), t1.name.strip()])
    return pair_list


def T_NamePairs2NameArray(pair_list):
    return [PAIR_SEP.join([dll_name, func_name]) if dll_name else func_name
            for dll_name, func_name in pair_list ]


def IMT_HASH(str):
    v = hashlib.md5(str).hexdigest()
    rv = int(v[:(1 + (FEATURE_MAX / 256)) * 2], 16) % FEATURE_MAX
    return rv


HASHDIC = {}


def addHASHDIC(hashv, name):
    if hashv in HASHDIC:
        nameset = HASHDIC[hashv]
        nameset.add(name)
    else:
        nameset = set([name])
        HASHDIC[hashv] = nameset


def T_NamePairs2FeatureVector(pair_list):
    features = [0] * (FEATURE_MAX + 1)

    if pair_list == IMP_CAN_NOT_OPEN_BY_PE_NONE:
        features[-1] = 1
    elif pair_list == IMP_CAN_OPEN_BUT_NONE:
        features[-1] = 0.5
    else:
        base = len(PRESET_DICT)
        rest = FEATURE_MAX - base
        for dll_name, func_name in pair_list:
            hashv = 0
            if func_name in PRESET_DICT.keys():
                hashv = PRESET_DICT[func_name]
            else:
                hashv = IMT_HASH(func_name) % rest + base

            if LEARNING_MODE == 1:
                addHASHDIC(hashv, func_name)

            features[hashv] = 1

    return features


def T_NameArray2NamePairs(name_list):
    pair_list = [name.split(PAIR_SEP, 1) for name in name_list]
    pair_list = [pair if len(pair) == 2 else ('', pair[0]) for pair in pair_list]
    return pair_list


def T_NameArray2FeatureVector(name_list):
    pair_list = T_NameArray2NamePairs(name_list)
    return T_NamePairs2FeatureVector(pair_list)


def T_NameArray2FeatureString(name_list):
    return COL_SEP.join([str(x) for x in T_NameArray2FeatureVector(name_list)])


def T_IndexArray2FeatureVector(sparsearr):
    result = [0] * FEATURE_MAX
    for item in sparsearr:
        result[item] = 1
    return COL_SEP.join(result)


def GetFeatureFromBuffer(input_buffer):
    pair_list = T_Buffer2NamePairs(input_buffer)
    name_list = T_NamePairs2NameArray(pair_list)
    return T_NameArray2FeatureString(name_list)


def GetFeatureFromFile(fname):
    with open(fname, 'rb') as f1:
        input_buffer = f1.read()
    return GetFeatureFromBuffer(input_buffer)


def GetMetaFromFile(fname):
    with open(fname, 'rb') as f1:
        input_buffer = f1.read()
    pair_list = T_Buffer2NamePairs(input_buffer)
    name_list = T_NamePairs2NameArray(pair_list)
    return COL_SEP.join(name_list)


def GetFeatureFromLine(line):
    arr = line.split(COL_SEP)
    return T_NameArray2FeatureString(arr)

'''
File -> Buffer -> NamePairs -> FeatureVector -> FeatureString
'''

def PARSE_SHA1_RAW_FILE_BYFOLDER(fstring):
    # PARSE SHA1-RAW
    flist = os.listdir(fstring)
    for sha1 in flist:
        if os.path.isfile(fstring + sha1):
            f1 = open(fstring + sha1)
            functions = f1.readlines()
            dataarr = [x.strip().strip('\0') for x in functions]
            print sha1 + COL_SEP + T_NameArray2FeatureString(dataarr)


def PARSE_SHA1_RAW_FILE(fname):
    # PARSE SHA1-RAW
    sha1 = fname.rsplit('/', 1)[-1].split('.', 1)[0]
    if os.path.isfile(fname):
        f1 = open(fname)
        functions = f1.readlines()
        dataarr = [x.strip().strip('\0') for x in functions]
        print sha1 + COL_SEP + T_NameArray2FeatureString(dataarr)


def PARSE_SINGLE_FLOG(fname):
    f1 = open(fname)
    while True:
        line = f1.readline()
        if not line:
            break
        if line.startswith('#'):
            continue
        z1 = line.strip().split(COL_SEP)
        sha1 = z1[0]
        dataarr = z1[1:]
        print sha1 + COL_SEP + T_NameArray2FeatureString(dataarr)
    f1.close()

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print 'Usage:' + sys.argv[0] + ' FLOG/PE/META FOLDER/FILE'
        exit()

    ACTION = sys.argv[1]
    TARGET = sys.argv[2]
    if ACTION == 'FLOG':
        if not os.path.isdir(TARGET):
            PARSE_SHA1_RAW_FILE(TARGET)
        else:
            PARSE_SHA1_RAW_FILE_BYFOLDER(TARGET)
    elif ACTION == 'PE':
        if not os.path.isdir(TARGET):
            print GetFeatureFromFile(TARGET)
        else:
            flist = os.listdir(TARGET)
            for f in flist:
                if os.path.isfile(TARGET + f):
                    print f + COL_SEP + GetFeatureFromFile(TARGET + f)
    elif ACTION == 'META':
        if not os.path.isdir(TARGET):
            print GetMetaFromFile(TARGET)
        else:
            flist = os.listdir(TARGET)
            for f in flist:
                if os.path.isfile(TARGET + f):
                    print f + COL_SEP + GetMetaFromFile(TARGET + f)

    bGenerateTable = True
    if bGenerateTable:
        f1 = open('__FunctionHashTable.table', 'a')
        for key in HASHDIC:
            nameset = list(HASHDIC[key])
            f1.write(str(key) + COL_SEP + COL_SEP.join(nameset) + '\n')
        f1.close()
