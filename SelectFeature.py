import os
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
from config import CFG
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import shuffle
import itertools
import seaborn as sns

result = []
for i in range(4):
    t_x = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['train_x']
    t_y = np.load(CFG.SVM.DATASET + 'fold' + str(i) + '.npz')['train_y']
    train_x, train_y = shuffle(t_x, t_y)
    x_y_result = mutual_info_classif(train_x,train_y)
    result.append(x_y_result)
    print(x_y_result)
rr = np.array(result)

index1 = pd.Series(np.arange(1, 25))
index1 = index1.astype(str)
index1 = index1

index2 = pd.Series(np.arange(1, 5))
index2 = index2.astype(str)
index2 = 'FOLD' + index2

value = rr.flatten()
value = pd.Series(value)
value.name = 'value'

prod = itertools.product(index2, index1)
prod = pd.DataFrame([x for x in prod])
prod.columns = ['FOLD', 'Feature']
prod = pd.concat([prod, value], axis=1)

sns.barplot(x='Feature', y='value', hue='FOLD', data=prod)
# plt.savefig('./energy_boxplot/S'+str(i+1)+'__C'+str(j*5+1)+'_'+str(j*5+6)+'.png')
plt.savefig('./figures/Feature_entropy_globnor.png')
plt.show()
print('over')