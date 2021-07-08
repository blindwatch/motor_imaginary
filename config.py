from easydict import EasyDict as edict

__C = edict()

CFG = __C

__C.RAWDATA_TRAIN = './data/raw/train/' #原始数据位置
__C.RAWDATA_TEST = './data/raw/test/'
__C.DATA_WAVED_TRAIN = './data/wavedata/train/' #小波变换后位置
__C.DATA_WAVED_TEST = './data/wavedata/test/'
__C.DATA_TEST = './data/test/'      
__C.DATA_TRAIN = './data/train/'    #训练集位置
__C.NET_NOW = 'bpnet'       #当前训练网络
__C.STATE = 'train'         #train or test

__C.DRSANET = edict()       #drsanet的相关设置
__C.DRSANET.NET = [24, 4, 4, 8, 16]
__C.DRSANET.WEIGHTS = './cache/drsa/'
__C.DRSANET.PREDICT = './predict/drsa/'
__C.DRSANET.TRAIN = edict()
__C.DRSANET.STRIDE = [2, 2, 2]
__C.DRSANET.TRAIN.DATASET = './data8300/train/' #从这里读取数据
__C.DRSANET.TRAIN.DATATEST = './data8300/test/'
__C.DRSANET.TRAIN.RESULT = './result/drsanet/train'
__C.DRSANET.TRAIN.NUM_ITERATION = 50
__C.DRSANET.TRAIN.BATCH_SIZE = 8
__C.DRSANET.TRAIN.SHUFFLE = True
__C.DRSANET.TRAIN.DEFAULT_LEARNING_RATE = 1e-3

__C.CANET = edict()
__C.CANET.WEIGHTS = './cache/canet/'    #权重缓存地址
__C.CANET.PREDICT = './predict/canet/'
__C.CANET.NET = [8, 8, 16, 32]
__C.CANET.TRAIN = edict()
__C.CANET.TRAIN.DATASET = './data8300/train/'
__C.CANET.TRAIN.DATATEST = './data8300/test/'
__C.CANET.TRAIN.RESULT = './result/canet/train/'
__C.CANET.TRAIN.NUM_ITERATION = 50
__C.CANET.TRAIN.BATCH_SIZE = 16
__C.CANET.TRAIN.SHUFFLE = True
__C.CANET.TRAIN.DEFAULT_LEARNING_RATE = 1e-3

__C.SVM = edict()
__C.SVM.RESULT = './result/svm/'
__C.SVM.DATASET = './data/train/'
__C.SVM.STATE = 'train'
__C.SVM.PREDICT = './predict/svm/'

__C.BP = edict()
__C.BP.WEIGHTS = './cache/bpnet/'
__C.BP.PREDICT = './predict/bp/'
__C.BP.NET = [24, 64, 2]
__C.BP.TRAIN = edict()
__C.BP.TRAIN.DATASET = './data/train/'
__C.BP.TRAIN.DATATEST = './data/test/'
__C.BP.TRAIN.RESULT = './result/bpnet/train/'
__C.BP.TRAIN.NUM_ITERATION = 200
__C.BP.TRAIN.BATCH_SIZE = 16
__C.BP.TRAIN.SHUFFLE = True
__C.BP.TRAIN.DEFAULT_LEARNING_RATE = 1e-4