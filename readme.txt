运行要求库：pytorch, numpy, sklearn, seaborn, pandas, matplotlib, pywt


config.py为设置网络结构、存储路径等内容。在config中可以设置train或者test，对应训练和输出结果。

neuralnetwork是神经网络训练的程序。

dataprocess.py 是数据集处理，CSP特征提取的程序

net.py 定义了神经网络的结构

svm.py是支持向量机的验证程序

l1csp.py是一范数CSP的优化程序（未在最后使用）

selectFeature.py绘制互信息图

result文件夹内有完整的实验数据


data文件夹中raw为原始数据，wavedata为小波变换之后数据（因为太大了，删掉了，可以通过wav.py生成）
，train为生成的24维向量，

data8300分别存放了提取的两种特征

predict为分类器预测的结果