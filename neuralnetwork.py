import os
import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
import random
from config import CFG
from net import *
from sklearn.utils import shuffle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False


class HmiDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

if CFG.NET_NOW == 'bpnet':
    cfg = CFG.BP
elif CFG.NET_NOW == 'canet':
    cfg = CFG.CANET
else:
    cfg = CFG.DRSANET

train_acc_s = [[] for _ in range(4)]
train_loss_s = [[] for _ in range(4)]
val_acc_s = [[] for _ in range(4)]
val_loss_s = [[] for _ in range(4)]

train_acc = [0, 0, 0, 0]
train_loss = [0, 0, 0,  0]
val_acc = [0, 0, 0, 0]
val_loss = [0, 0, 0, 0]
best = []

def cross_validation():
    count = 0
    for i in range(4):
        t_x = np.load(cfg.TRAIN.DATASET + 'fold' + str(i) + '.npz')['train_x'].astype(np.float32)
        t_y = np.load(cfg.TRAIN.DATASET + 'fold' + str(i) + '.npz')['train_y']
        val_x = np.load(cfg.TRAIN.DATASET + 'fold' + str(i) + '.npz')['val_x'].astype(np.float32)
        val_y = np.load(cfg.TRAIN.DATASET + 'fold' + str(i) + '.npz')['val_y']
        train_x, train_y = shuffle(t_x, t_y)
        train_set = HmiDataset(train_x, train_y)
        val_set = HmiDataset(val_x, val_y)
        fold_start_time = time.time()
        best.append(train(train_set, val_set, count))
        print('[%03d/5]FOLD %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
              (count + 1, time.time() - fold_start_time, train_acc_s[int(count)][-1],
               train_loss_s[int(count)][-1],
               val_acc_s[int(count)][-1], val_loss_s[int(count)][-1]))
        count = count + 1


def train(train_set, val_set, count):
    vmin_loss = 10
    device = torch.device('cuda:0')
    if CFG.NET_NOW == 'bpnet':
        model = BPnet(cfg.NET)
    elif CFG.NET_NOW == 'canet':
        model = CANet(cfg.NET)
    else:
        model = DRSANet(cfg.NET, cfg.STRIDE)
    model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.DEFAULT_LEARNING_RATE, eps=1e-8, weight_decay=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.DEFAULT_LEARNING_RATE, momentum=0.9, weight_decay=1e-3)
    random.seed(1)
    loss = nn.CrossEntropyLoss()
    epoch = 0
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    best_model = [1, 0]
    while epoch < cfg.TRAIN.NUM_ITERATION:
        epoch_start_time = time.time()
        t_acc = 0.0
        t_loss = 0.0
        v_acc = 0.0
        v_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda().squeeze())
            batch_loss.backward()
            optimizer.step()

            t_loss += batch_loss.item()
            t_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze())

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda().squeeze())

                v_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze())
                v_loss += batch_loss.item()

        train_acc_s[count].append(t_acc / train_set.__len__())
        train_loss_s[count].append(t_loss / train_set.__len__())
        val_acc_s[count].append(v_acc / val_set.__len__())
        val_loss_s[count].append(v_loss / val_set.__len__())
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
              (epoch + 1, cfg.TRAIN.NUM_ITERATION, time.time() - epoch_start_time, train_acc_s[int(count)][-1],
               train_loss_s[int(count)][-1],
               val_acc_s[int(count)][-1], val_loss_s[int(count)][-1]))
        epoch = epoch + 1
        if val_loss_s[int(count)][-1] < vmin_loss:
            vmin_loss = val_loss_s[int(count)][-1]
            train_acc[count] = train_acc_s[int(count)][-1]
            train_loss[count] = train_loss_s[int(count)][-1]
            val_acc[count] = val_acc_s[int(count)][-1]
            val_loss[count] = val_loss_s[int(count)][-1]
        if val_loss_s[int(count)][-1] < best_model[0] and val_acc_s[int(count)][-1] > best_model[1]:
            best_model[0] = val_loss_s[int(count)][-1]
            best_model[1] = val_acc_s[int(count)][-1]
            print([best_model[0], best_model[1]])
            if not os.path.exists(cfg.WEIGHTS):
                os.makedirs(cfg.WEIGHTS)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, cfg.WEIGHTS+'fold' + str(count) + '.pth')
    return best_model

if CFG.STATE == 'train':
    cross_validation()
    if not os.path.exists(cfg.TRAIN.RESULT):
        os.makedirs(cfg.TRAIN.RESULT)

    if CFG.NET_NOW == 'bpnet':
        for i in range(len(train_acc_s)):
            train_acc[i] = np.mean(train_acc_s[i][50:])
            train_loss[i] = np.mean(train_loss_s[i][50:])
            val_acc[i] = np.mean(val_acc_s[i][50:])
            val_loss[i] = np.mean(val_loss_s[i][50:])

    x = np.linspace(1, len(train_acc_s[0]), num=len(train_acc_s[0]))
    num = 0
    for i in range(len(train_acc_s)):
        num += 1
        plt.plot(x, train_acc_s[i], 'o-', label = '第'+ str(num) + '次')
    plt.title('交叉训练集正确率')
    plt.legend()
    plt.savefig(cfg.TRAIN.RESULT + 'cross_tacc.png')
    plt.xticks(np.linspace(0, len(train_acc_s[0]), num=11))
    plt.yticks(np.linspace(0.4, 1, num=12))
    plt.show()

    num = 0
    for i in range(len(train_acc_s)):
        num += 1
        plt.plot(x, train_loss_s[i], 'o-', label='第' + str(num) + '次')
    plt.title('交叉训练集损失')
    plt.legend()
    plt.savefig(cfg.TRAIN.RESULT + 'cross_tloss.png')
    plt.xticks(np.linspace(0, len(train_acc_s[0]), num=11))
    plt.show()

    num = 0
    for i in range(len(train_acc_s)):
        num += 1
        plt.plot(x, val_acc_s[i], 'o-', label = '第'+ str(num) + '次')
    plt.title('交叉验证集正确率')
    plt.legend()
    plt.savefig(cfg.TRAIN.RESULT + 'cross_vacc.png')
    plt.xticks(np.linspace(0, len(train_acc_s[0]), num=11))
    plt.yticks(np.linspace(0.4, 1, num=12))
    plt.show()

    num = 0
    for i in range(len(train_acc_s)):
        num += 1
        plt.plot(x, val_loss_s[i], 'o-', label = '第'+ str(num) + '次')
    plt.title('交叉验证集损失')
    plt.legend()
    plt.savefig(cfg.TRAIN.RESULT + 'cross_vloss.png')
    plt.xticks(np.linspace(0, len(train_acc_s[0]), num=11))
    plt.show()

    xx = np.linspace(1, 4, 4)
    plt.plot(xx, train_acc, 'o-')
    plt.title('训练正确率')
    plt.savefig(cfg.TRAIN.RESULT + 'train_acc.png')
    plt.xticks(np.linspace(1, 4, 4))
    plt.show()

    plt.plot(xx, val_acc, 'o-')
    plt.title('测试正确率')
    plt.savefig(cfg.TRAIN.RESULT + 'val_acc.png')
    plt.xticks(np.linspace(1, 4, 4))
    plt.show()

    fold_acc_t = pd.DataFrame(data=train_acc_s, index=['1', '2', '3', '4'])
    fold_acc_t.to_csv(cfg.TRAIN.RESULT + 'fold_acc_t.csv')
    fold_loss_t = pd.DataFrame(data=train_loss_s, index=['1', '2', '3', '4'])
    fold_loss_t.to_csv(cfg.TRAIN.RESULT + 'fold_loss_t.csv')
    fold_acc_v = pd.DataFrame(data=val_acc_s, index=['1', '2', '3', '4'])
    fold_acc_v.to_csv(cfg.TRAIN.RESULT + 'fold_acc_v.csv')
    fold_loss_v = pd.DataFrame(data=val_loss_s, index=['1', '2', '3', '4'])
    fold_loss_v.to_csv(cfg.TRAIN.RESULT + 'fold_loss_v.csv')

    acc = pd.DataFrame(data=train_acc, columns=['train_acc'])
    acc.to_csv(cfg.TRAIN.RESULT + 'train_acc.csv')
    val = pd.DataFrame(data=val_acc, columns=['val_acc'])
    val.to_csv(cfg.TRAIN.RESULT + 'val_acc.csv')

    best_fold = np.array(best)
    best_df = pd.DataFrame(data=best_fold, columns=['loss', 'acc'])
    best_df.to_csv(cfg.TRAIN.RESULT + 'fold_result.csv')

if not os.path.exists(cfg.PREDICT):
    os.makedirs(cfg.PREDICT)

if CFG.STATE == 'test':
    testpath = cfg.TRAIN.DATATEST
    names = os.listdir(testpath)
    df_weights = pd.read_csv(cfg.TRAIN.RESULT + 'fold_result.csv')['acc']
    df_weights = df_weights / np.sum(df_weights)
    for i, name in enumerate(names):
        result=np.zeros((200, 5))
        x = np.load(testpath + name)['fold'].astype(np.float32)
        for j in range(len(x)):
            val_x = x[j]
            if CFG.NET_NOW == 'bpnet':
                model = BPnet(cfg.NET)
            elif CFG.NET_NOW == 'canet':
                model = CANet(cfg.NET)
            else:
                model = DRSANet(cfg.NET, cfg.STRIDE)
            # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.DEFAULT_LEARNING_RATE, eps=1e-8, weight_decay=1e-3)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.DEFAULT_LEARNING_RATE, momentum=0.9,weight_decay=1e-3)
            checkpoint = torch.load(cfg.WEIGHTS+'fold' + str(j) + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.eval()
            test_set = HmiDataset(val_x)
            test_loader = DataLoader(test_set, batch_size=val_x.shape[0], shuffle=False)
            with torch.no_grad():
                for data in test_loader:
                    val_pred = model(data)
                    label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
                    result[:, j:j+1] = label.reshape(-1, 1)
        for j in range(len(result)):
            vote = 0
            for p in range(len(x)):
                vote += df_weights[p] * result[j, p]
            if vote > 0.5:
                result[j, -1] = 1
            else:
                result[j, -1] = 0
        df = pd.DataFrame(result, columns=['fold1', 'fold2', 'fold3', 'fold4', 'bagresult'])
        df.to_csv(cfg.PREDICT + 's' + str(i+5) + '.csv')


