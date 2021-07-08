import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False

train_comp = pd.read_csv("./train_acc_compare.csv")
val_comp = pd.read_csv("./val_acc_compare.csv")
t_choose = pd.read_csv("./ttdim_choose.csv")
v_choose = pd.read_csv("./tvdim_choose.csv")



plt.figure(dpi=150)
sns.stripplot(data=train_comp, jitter=True , size=7, palette="Set2", marker="D",edgecolor="gray")
sns.boxplot(data=train_comp, whis=5)
plt.title('4FOLDS训练集正确率')
plt.savefig('train_comp.png')
plt.show()

plt.figure(dpi=150)
sns.stripplot(data=val_comp, jitter=True , size=7, palette="Set2", marker="D",edgecolor="gray")
sns.boxplot(data=val_comp, whis=5)
plt.title('4FOLDS验证集正确率')
plt.savefig('val_comp.png')
plt.show()

plt.figure(dpi=150)
sns.stripplot(data=t_choose, jitter=True , size=7, palette="Set2", marker="D",edgecolor="gray")
sns.boxplot(data=t_choose, whis=5)
plt.title('4FOLDS训练集正确率')
plt.savefig('t_choose.png')
plt.show()

plt.figure(dpi=150)
sns.stripplot(data=v_choose, jitter=True , size=7, palette="Set2", marker="D",edgecolor="gray")
sns.boxplot(data=v_choose, whis=5)
plt.title('4FOLDS测试集正确率')
plt.savefig('v_choose.png')
plt.show()