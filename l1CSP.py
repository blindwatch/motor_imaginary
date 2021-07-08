import numpy as np
import copy
from matplotlib import pyplot as plt
import time

class Adam():       #adam优化器
    def __init__(self,  beta1=0.9, beta2=0.999, epislon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.m = 0
        self.v = 0
        self.t = 0

    def minimize(self, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m_cat = self.m / (1 - self.beta1 ** self.t)
        self.v_cat = self.v / (1 - self.beta2 ** self.t)
        return self.m_cat / (self.v_cat ** 0.5 + self.epislon)


class L1CSP():      #复现了L1-CSP的论文代码
    def __init__(self, limit, lr, th):
        self.limit = limit
        self.lr = lr
        self.th = th

    def __call__(self, x, y, outnum):
        w = np.zeros((x.shape[0], 1))
        index = np.argmax(np.sum(np.power(x, 2), axis=1))
        w[:, 0:1] = x[:, index:index+1]
        w = w / np.linalg.norm(w)
        X = np.copy(x)
        Y = np.copy(y)
        v = []
        s = []
        w_st = time.time()
        for epoch in range(self.limit):
            optimizer = Adam()
            w_t = w[:, 0:1].copy()
            leftt = w_t.T.dot(x)
            left_b = np.abs(leftt)
            left = np.sum((leftt / left_b) * x, axis=1)
            left_bb = np.sum(left_b)
            rightt = w_t.T.dot(y)
            right_b = np.abs(rightt)
            right = np.sum((rightt / right_b) * y, axis=1)
            right_bb = np.sum(right_b)
            dt = left / np.sum(left_b) - right / right_bb
            score = left_bb / right_bb
            s.append(score)
            if len(s) > 1:
                v.append(s[-1] - s[-2])
            nwscore = score
            jb = 0
            for j in self.lr:
                nwmax = w_t + j * dt.reshape(-1, 1)
                ns = np.sum(np.abs(nwmax.T.dot(X))) / np.sum(np.abs(nwmax.T.dot(Y)))
                if ns > nwscore:
                    w[:, 0:1] = nwmax / np.linalg.norm(nwmax)
                    jb = j
                    nwscore = ns
            if len(v) > 5:
                if (np.array([v[-1], v[-2], v[-3], v[-4], v[-5]]).all() <= self.th).all():
                    print(epoch)
                    break
            print([nwscore, jb])
        w[:, 0:1] = w[:, 0:1] / np.linalg.norm(w[:, 0:1])
        print('ftime:{}'.format(time.time()-w_st))

        for j in range(1, outnum):
            s = []
            v = []
            b = np.eye(X.shape[0]) - w[:, j - 1].dot(w[:, j - 1].T)
            X = b.dot(X)
            Y = b.dot(Y)
            ak = np.ones((X.shape[0], 1))
            ak = ak / np.linalg.norm(ak)
            w_st=time.time()
            for epoch in range(self.limit):
                optimizer = Adam()
                leftt = ak.T.dot(X)
                left_b = np.abs(leftt)
                left = np.sum((leftt / left_b) * X, axis=1)
                left_bb = np.sum(left_b)
                rightt = ak.T.dot(Y)
                right_b = np.abs(rightt)
                right = np.sum((rightt / right_b) * Y, axis=1)
                right_bb = np.sum(right_b)
                dt = left / np.sum(left_b) - right / right_bb
                score = left_bb / right_bb
                s.append(score)
                if len(s) > 1:
                    v.append(s[-1] - s[-2])
                nwscore = score
                jb=0
                for p in self.lr:
                    nwmax = ak + p * dt.reshape(-1, 1)
                    ns = np.sum(np.abs(nwmax.T.dot(X))) / np.sum(np.abs(nwmax.T.dot(Y)))
                    if ns > nwscore:
                        ak[:, 0:1] = nwmax / np.linalg.norm(nwmax)
                        nwscore = ns
                        jb=p
                if len(v) > 5:
                    if (np.array([v[-1], v[-2], v[-3], v[-4], v[-5]]) <= self.th).all():
                        print(epoch)
                        break
                print([nwscore, jb])
            ak[:, 0:1] = ak[:, 0:1] / np.linalg.norm(ak[:, 0:1])
            shift = np.eye(X.shape[0]) - w.dot(w.T)
            wk1 = np.zeros((X.shape[0], 1))
            wk1[:, 0:1] = shift.dot(ak)
            wk1[:, 0:1] = wk1[:, 0:1] / np.linalg.norm(wk1)
            w = np.concatenate((w, wk1), axis=1)
            print('ftime:{}'.format(time.time() - w_st))
        return w.T
