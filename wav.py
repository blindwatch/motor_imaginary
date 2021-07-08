import pywt
import numpy as np
from config import CFG
import matplotlib.pyplot as plt
import os



#两个节律
iter_freqs = [
    {'name': 'w1', 'fmin': 7, 'fmax': 12},
    {'name': 'w2', 'fmin': 12, 'fmax': 16},
    {'name': 'w3', 'fmin': 16, 'fmax': 20},
    {'name': 'w4', 'fmin': 18, 'fmax': 26},
]


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def wave_norm(x):       #归一化
    mean = np.mean(x, axis=0, keepdims=True)
    max = np.max(mean, axis=2, keepdims=True)
    min = np.min(mean, axis=2, keepdims=True)
    wav_x = -1 + 2*(x - min)/(max - min)
    return wav_x
'''
def wave_norm(x):
    mean = np.mean(x, axis=0)
    max = mean.max()
    min = mean.min()
    wav_x = -1 + 2*(x - min)/(max - min)
    return wav_x
'''


def wave_regular(x):        #zscore
    mean = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(mean, axis=2, keepdims=True, ddof=1)
    mu = np.mean(mean, axis=2, keepdims=True)
    wav_x = (x-mu) / sigma
    return wav_x

'''
def wave_regular(x):
    #mean = np.mean(x, axis=0, keepdims=True)
    sigma = x.std(ddof=1)
    mu = x.mean()
    wav_x = (x-mu) / sigma
    return wav_x
'''

def TimeFrequencyWP(data, fs, wavelet, maxlevel=4): #小波分解重构提取
    dt = {}
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** (maxlevel))
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    #fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(10, 12), sharex=True, sharey=False)
    # 绘制原始数据
    #axes[0].plot(data)
    #axes[0].set_title('原始数据')

    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 绘制对应频率的数据
        #axes[iter + 1].plot(new_wp.reconstruct(update=True))
        # 设置图名
        #axes[iter + 1].set_title(iter_freqs[iter]['name'])
        dt[iter_freqs[iter]['name']] = new_wp.reconstruct(update=True)
    #plt.show()
    return dt

if not os.path.exists(CFG.DATA_WAVED_TRAIN):
    os.makedirs(CFG.DATA_WAVED_TRAIN)
    os.makedirs(CFG.DATA_WAVED_TEST)

'''#训练集的波形提取
names = os.listdir(CFG.RAWDATA_TRAIN)
for d, name in enumerate(names):
    x = np.load(CFG.RAWDATA_TRAIN + name)['X']
    y = np.load(CFG.RAWDATA_TRAIN + name)['y']
    #x = wave_regular(x)
    w_list=[]
    for i in range(len(iter_freqs)):
        w_list.append(np.zeros((200, 59, 300)))
    for i in range(200):
        for j in range(59):
            dt = TimeFrequencyWP(x[i, j, :], 50, wavelet='db4', maxlevel=6)
            for k, wname in enumerate(dt.keys()):
                w_list[k][i, j] = dt[wname][0:300]
        print(i)
    #for k in range(len(w_list)):
    #     w_list[k] = wave_regular(w_list[k])
    print(name)
    np.savez(CFG.DATA_WAVED_TRAIN + name, X=w_list[0], X_1=w_list[1], X_2=w_list[2], X_3=w_list[3], y=y)


'''
#测试集的波形提取
names = os.listdir(CFG.RAWDATA_TEST)
for d, name  in enumerate(names):
    x = np.load(CFG.RAWDATA_TEST + name)['X']
    w_list=[]
    for i in range(len(iter_freqs)):
        w_list.append(np.zeros((200, 59, 300)))
    for i in range(200):
        for j in range(59):
            dt = TimeFrequencyWP(x[i, j, :], 50, wavelet='db4', maxlevel=6)
            for k, wname in enumerate(dt.keys()):
                w_list[k][i, j] = dt[wname][0:300]
        print(i)
    for k in range(len(w_list)):
        w_list[k] = wave_regular(w_list[k])
    np.savez(CFG.DATA_WAVED_TEST + name, X=w_list[0], X_1=w_list[1], X_2=w_list[2], X_3=w_list[3])



