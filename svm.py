import os
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
from config import CFG
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False

cc = 10
ga = 0.001

train_acc = []
val_acc = []
foldclassifier = []
foldweight = []
count = 0
for i in range(4):
    t_x = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['train_x']
    t_y = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['train_y']
    val_x = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['val_x']
    val_y = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['val_y']
    train_x, train_y = shuffle(t_x, t_y)
    fold_start_time = time.time()
    '''
    lda = LinearDiscriminantAnalysis(solver='eigen', n_components=1,shrinkage='auto')
    lda.fit(train_x, train_y)
    train_lda_X = lda.transform(train_x)
    val_lda_x = lda.transform(val_x)
    train_xxl = np.where(train_y == 0)[0]
    train_xxr = np.where(train_y == 1)[0]
    plt.plot(train_x[train_xxl], 'x', label="L", color='blue')
    plt.plot(train_x[train_xxr], 'o', label="r", color='green')
    plt.show()

    val_xxl = np.where(val_y == 0)[0]
    val_xxr = np.where(val_y == 1)[0]
    plt.plot(val_x[val_xxl], 'x', label="L", color='yellow')
    plt.plot(val_x[val_xxr], 'o', label="r", color='red')
    plt.show()
    
    train_acc.append(lda.score(train_x, train_y))
    val_acc.append(lda.score(val_x, val_y))
    '''
    classifier = svm.SVC(C=cc, kernel='rbf', gamma=ga, decision_function_shape='ovr')
    classifier.fit(train_x, train_y.ravel())
    foldclassifier.append(classifier)
    foldweight.append(classifier.score(val_x, val_y))
    train_acc.append(classifier.score(train_x, train_y))
    val_acc.append(classifier.score(val_x, val_y))
    print('[%03d/5]FOLD %2.2f sec(s) Train Acc: %3.6f  | Val Acc: %3.6f ' %
          (count + 1, time.time() - fold_start_time, train_acc[-1], val_acc[-1]))
    count = count + 1

if not os.path.exists(CFG.SVM.RESULT):
    os.makedirs(CFG.SVM.RESULT)

if CFG.SVM.STATE == 'train':
    plt.plot(train_acc)
    plt.title('训练正确率')
    plt.savefig(CFG.SVM.RESULT + 'train_acc.png')
    plt.show()

    plt.plot(val_acc)
    plt.title('测试正确率')
    plt.savefig(CFG.SVM.RESULT + 'val_acc.png')
    plt.show()

    acc = pd.DataFrame(data=train_acc, columns=['train_acc'])
    acc.to_csv(CFG.SVM.RESULT + 'train_acc.csv')
    val = pd.DataFrame(data=val_acc, columns=['val_acc'])
    val.to_csv(CFG.SVM.RESULT + 'val_acc.csv')


if not os.path.exists(CFG.SVM.PREDICT):
    os.makedirs(CFG.SVM.PREDICT)

if CFG.SVM.STATE == 'test':
    path = CFG.DATA_TEST
    names = os.listdir(path)
    foldweights = np.array(foldweight)
    foldweights = foldweights / np.sum(foldweights)
    for i, name in enumerate(names):
        result=np.zeros((200, 5))
        x = np.load(path + name)['fold']
        for j in range(len(x)):
            classifierNow = foldclassifier[j]
            val_x = x[j]
            result[:, j:j+1] = classifierNow.predict(val_x).reshape(-1, 1)
        for j in range(len(result)):
            vote = 0
            for p in range(len(x)):
                vote += foldweights[p] * result[j, p]
            if vote > 0.5:
                result[j, -1] = 1
            else:
                result[j, -1] = 0
        df = pd.DataFrame(result, columns=['fold1', 'fold2', 'fold3', 'fold4', 'bagresult'])
        df.to_csv(CFG.SVM.PREDICT + 's' + str(i+5) + '.csv')





