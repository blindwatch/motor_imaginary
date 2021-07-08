import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import random
from config import CFG
from l1CSP import L1CSP
from sklearn.model_selection import StratifiedKFold

plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False

FeatureL = 3        #要提取的特征数 L*2
control = 0         #control0提取一维特征，control1提取二维的8n*300特征

def readfile(path):
    names = os.listdir(path)
    x = []
    y = []
    for i, name in enumerate(names):
        x.append(np.load(path + name))
        y.append(np.load(path + name)['y'])
    index = x[-1].files[:-1]
    return x, y, index

def CSP(data, label):

    #瑞利熵求解
    channel = np.size(data, 1)
    trials = np.size(data, 0)
    n_class = len(set(label.flatten()))

    cov_mat = np.zeros((trials, channel, channel), dtype=np.float32)
    for i in range(trials):
        e = data[i, :, :]
        cov = e.dot(e.transpose())
        cov_mat[i] = cov / cov.trace()

    labeled_cov = {}
    for i in range(n_class):
        labeled_cov[i] = np.mean(cov_mat[np.where(label == i)[0]], axis=0)


    cov_total1 = np.linalg.inv(labeled_cov[1]).dot(labeled_cov[0])
    Dt1, Ut1 = np.linalg.eig(cov_total1)
    dtindex1 = np.argsort(np.abs(Dt1))[::-1]
    Ut1 = Ut1[:, dtindex1]
    M1 = Ut1.T[0:FeatureL, :]


    cov_total2 = np.linalg.inv(labeled_cov[0]).dot(labeled_cov[1])
    Dt2, Ut2 = np.linalg.eig(cov_total2)
    dtindex2 = np.argsort(np.abs(Dt2))[::-1]
    Ut2 = Ut2[:, dtindex2]
    M2 = Ut2.T[0:FeatureL, :]
    CSPMatrix = np.concatenate((M1, M2), axis=0)


    '''#RD正则化
    labeled_mean = {}
    for i in range(n_class):
        labeled_mean[i] = np.mean(data[np.where(label == i)[0]], axis=0)
    channel_cov = labeled_mean[0].dot(labeled_mean[1].T)
    RD1 = np.diag(np.sum(np.abs(channel_cov), axis=1) / channel)
    RD1 = RD1 / RD1.max()
    RD2 = np.diag(np.sum(np.abs(channel_cov), axis=0) / channel)
    RD2 = RD2 / RD2.max()
    '''
    '''#白化求解
    cov_mean_total = labeled_cov[0] + labeled_cov[1]
    Dt,Ut = np.linalg.eig(cov_mean_total)
    dtindex = np.argsort(np.abs(Dt))[::-1]
    Dt = Dt[dtindex]
    Ut = Ut[:, dtindex]
    P = np.dot(np.diag(np.sqrt(1 / Dt)), Ut.transpose())
    transCov1 = P.dot(labeled_cov[0]).dot(P.transpose())

    D1, U1 = np.linalg.eig(transCov1)
    d1index = np.argsort(np.abs(D1))[::-1]
    U1 = U1[:, d1index]
    CSPMatrix = U1.T.dot(P)
    D1 = np.diag(D1[d1index])
    '''
    '''L1CSP 直接优化
    x1 = data[np.where(label == 0)[0]].transpose(1, 0, 2).reshape(data.shape[1], -1)
    x2 = data[np.where(label == 1)[0]].transpose(1, 0, 2).reshape(data.shape[1], -1)
    csp = L1CSP(500, [10, 1, 0.5, 0.2, 1e-1, 0.05, 0.02, 1e-2, 1e-3], 1e-4)
    m1 = csp(x1, x2, FeatureL)
    m2 = csp(x2, x1, FeatureL)
    CSPMatrix = np.concatenate((m1, m2), axis=0)
    '''
    '''L1CSP bootstrap优化
    x1 = data[np.where(label == 0)[0]]
    x2 = data[np.where(label == 1)[0]]
    m1 = []
    m2 = []
    csp = L1CSP(500, [0.05, 0.02, 1e-2, 1e-3, 0.2, 1e-1, 1, 2, 5, 10, 100, 400], 0.1)
    for i in range(5):
        index1 = np.random.choice(np.arange(x1.shape[0]), size=int(x1.shape[0] / 10), replace=False)
        index2 = np.random.choice(np.arange(x2.shape[0]), size=int(x2.shape[0] / 10), replace=False)
        xx1 = x1[index1].transpose(1, 0, 2).reshape(x1.shape[1], -1)
        xx2 = x2[index2].transpose(1, 0, 2).reshape(x2.shape[1], -1)
        m1.append(csp(xx1, xx2, FeatureL))
        m2.append(csp(xx2, xx1, FeatureL))
    mm1 = np.mean(np.array(m1), axis=0)
    mm2 = np.mean(np.array(m2), axis=0)
    CSPMatrix = np.concatenate((mm1, mm2), axis=0)
    '''
    return CSPMatrix


def FeatureExtra(CspMat, data):
    global control
    trials = np.size(data, 0)
    Filter = np.r_[CspMat[:FeatureL], CspMat[-FeatureL:]]
    if control==1:
        samples = np.size(data, 2)
        feature = np.zeros([trials, 2 * FeatureL, samples])
        for t in range(trials):
            project = Filter.dot(data[t])
            feature[t, :, :] = np.log(np.abs(project))
        return feature

    feature = np.zeros([trials, 2 * FeatureL])
    #L2
    for t in range(trials):
        project = Filter.dot(data[t])
        variances = np.var(project, 1)
        for f in range(len(variances)):
            feature[t, f] = np.log(variances[f] / np.sum(variances))
    return feature
    '''
    #L1
    for t in range(trials):
        project = Filter.dot(data[t])
        for f in range(project.shape[0]):
            feature[t, f] = np.linalg.norm(project[f], ord=1)
    return feature
    '''

if not os.path.exists(CFG.CANET.TRAIN.DATASET):
    os.makedirs(CFG.CANET.TRAIN.DATASET)
if not os.path.exists(CFG.SVM.DATASET):
    os.makedirs(CFG.SVM.DATASET)


cspMatrix = []      #训练集特征提取
#cspMatrix = np.load('cspmatrix.npz')['cspma']
rawx, ylabel, wave_list = readfile(CFG.DATA_WAVED_TRAIN)
for i in range(len(ylabel)):
    val_x = {}
    val_y = ylabel[i]
    train_x = {}
    train_y = np.concatenate((np.array(ylabel)[np.delete(np.arange(len(ylabel)), i), :]))
    Feature_t = []
    Feature_v = []
    for k, j in enumerate(wave_list):
        val_x[j] = rawx[i][j]
        train_index = np.delete(np.arange(len(ylabel)), i)
        train_x[j] = rawx[train_index[0]][j]
        for p in range(1, len(train_index)):
            temp = rawx[train_index[p]][j].copy()
            train_x[j] = np.concatenate((train_x[j], temp), axis=0)
        CSPM = CSP(train_x[j], train_y)
        cspMatrix.append(CSPM)
        #CSPM = cspMatrix[i*4+k]
        Feature_t.append(FeatureExtra(CSPM, train_x[j]))
        Feature_v.append(FeatureExtra(CSPM, val_x[j]))

        '''
        train_xxl = np.where(train_y == 0)[0]
        train_xxr = np.where(train_y == 1)[0]
        plt.plot(Feature_t[k][train_xxl][:, 0], Feature_t[k][train_xxl][:, 3], 'x', label="L", color='blue')
        plt.plot(Feature_t[k][train_xxr][:, 0], Feature_t[k][train_xxr][:, 3], 'o', label="r", color='green')
        plt.show()

        train_xl = np.where(val_y == 0)[0]
        train_xr = np.where(val_y == 1)[0]
        plt.plot(Feature_v[k][train_xl][:, 0], Feature_v[k][train_xl][:, 3], 'x', label="L", color='coral')
        plt.plot(Feature_v[k][train_xr][:, 0], Feature_v[k][train_xr][:, 3], 'o', label="r", color='yellow')
        plt.show()
        '''

    featrue_t = np.array(Feature_t)
    featrue_v = np.array(Feature_v)
    if control == 0:
        feature_t = featrue_t.transpose((1, 0, 2)).reshape(-1, featrue_t.shape[0] * featrue_t.shape[2])
        feature_v = featrue_v.transpose((1, 0, 2)).reshape(-1, featrue_v.shape[0] * featrue_v.shape[2])
        tmu = np.mean(feature_t, axis=0)
        tsigma = np.mean(feature_t, axis=0)
        feature_t = (feature_t - tmu) / tsigma
        vmu = np.mean(feature_v, axis=0)
        vsigma = np.mean(feature_v, axis=0)
        feature_v = (feature_v - vmu) / vsigma
        np.savez(CFG.SVM.DATASET + "fold" + str(i) + ".npz", train_x=feature_t, train_y=train_y, val_x=feature_v,
                     val_y=val_y)
    else:
        feature_t = featrue_t.transpose((1, 0, 2, 3)).reshape((featrue_t.shape[1], featrue_t.shape[0] * featrue_t.shape[2], featrue_t.shape[3]))
        feature_v = featrue_v.transpose((1, 0, 2, 3)).reshape((featrue_v.shape[1], featrue_v.shape[0] * featrue_v.shape[2], featrue_v.shape[3]))
        np.savez(CFG.CANET.TRAIN.DATASET + "fold"+str(i)+".npz", train_x=feature_t, train_y=train_y, val_x=feature_v, val_y=val_y)
np.savez("cspmatrix.npz", cspma=np.array(cspMatrix))
print('over')






'''测试数据特征提取
if not os.path.exists(CFG.CANET.TRAIN.DATATEST):
    os.makedirs(CFG.CANET.TRAIN.DATATEST)
if not os.path.exists(CFG.DATA_TEST):
    os.makedirs(CFG.DATA_TEST)

cspMatrix = np.load('cspmatrix.npz')['cspma']
path = CFG.DATA_WAVED_TEST
names = os.listdir(path)
for i, name in enumerate(names):
    x = np.load(path+name)
    index = x.files
    Fold = []
    for p in range(4):
        Feature = []
        xout ={}
        for k, j in enumerate(index):
            xout[j] = x[j]
            #mean = np.mean(xout[j], axis=2, keepdims=True)
            #xout[j] = xout[j] - np.mean(xout[j], axis=2, keepdims=True)
            #for p in range(len(xout[j])):
            #    xout[j][p] = xout[j][p] / np.sqrt(xout[j][p].dot(xout[j][p].T).trace())
            Feature.append(FeatureExtra(cspMatrix[p*4 + k], xout[j]))
        featrue = np.array(Feature)
        if control == 0:
            feature = featrue.transpose((1, 0, 2)).reshape(-1, featrue.shape[0] * featrue.shape[2])
            mu = np.mean(feature, axis=0)
            sigma = np.mean(feature, axis=0)
            feature = (feature - mu) / sigma
        else:
            feature = featrue.transpose((1, 0, 2, 3)).reshape(
                (featrue.shape[1], featrue.shape[0] * featrue.shape[2], featrue.shape[3]))
        Fold.append(feature)
    if control == 0:
        np.savez(CFG.DATA_TEST + "n"+str(i+5)+".npz", fold=Fold)
    else:
        np.savez(CFG.CANET.TRAIN.DATATEST + "n" + str(i + 5) + ".npz", fold=Fold)
'''




