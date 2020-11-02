from collections import defaultdict
import numpy as np
import copy
import time
import sys
import json
import random
import csv
import socket
import math
from multiprocessing import Pool
from functools import partial
from help import send,receive
from sklearn import preprocessing
from phe import paillier
from phe import EncryptedNumber
import config
import random 

'''
Modified methods:
    find_min_dis_batch
'''


data_path = config._DATA_PATH
dim = -1
def load_data():
    global dim
    points = np.loadtxt(data_path)
    dim = points.shape[1]
    return points
batch_size=config._BATCH_SIZE
K=None
cpu_number=config._S1_CPU_NUMBER

class KMEANS:
    def __init__(self, n_cluster, epsilon=config._EQUAL_TOL, maxstep=config._MAX_STEP):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.N = None
        self.centers = None
        self.server = None
        self.pubkey, self.prikey = paillier.generate_paillier_keypair()
        self.cluster = defaultdict(list)

    def initial_0(self,data):
        # 分发密钥
        t1_start = time.perf_counter()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip = config._S2_IP
        port = config._S2_PORT

        server.connect((ip, port))
        self.server = server
        send(server, self.prikey)
        print("分发密钥的时间：")
        print(time.perf_counter() - t1_start)
        self.init_center(data)
        return
    def init_center(self,data):
        np.random.seed(1)
        self.N = data.shape[0]
        random_ind = np.random.choice(self.N, size=self.n_cluster)
        # random_ind = [500, 1200, 2000, 2500]
        self.centers = [data[i] for i in random_ind]
        return
    def enc(self,number):
        return self.pubkey.encrypt(number)

    def initial_1(self, data):
        csh_start = time.perf_counter()
        self.initial_0(data)
        ci_enc_list = []
        pool = Pool(cpu_number)
        centers=np.array(self.centers).flatten()
        ci_enc_list=pool.map(self.enc, centers)
        Cdata_enc = np.array(ci_enc_list).reshape(K,dim)
        data=np.array(data).flatten()
        t3_start = time.perf_counter()
        try:
            data_enc=pool.map(self.enc, data)
        except Exception as e:
            print(data)
        print("加密时间:")
        print(time.perf_counter() - t3_start)
        self.centers = Cdata_enc
        pool.close()
        pool.join()
        return np.array(data_enc).reshape(config._NUMBER, dim)
    def cal_dis1_one(self, data_enc_point):
        Bi = data_enc_point - self.centers

        hx_Bi = np.empty((Bi.shape[0], Bi.shape[1]), dtype=EncryptedNumber)
        eps_Bi = np.empty((Bi.shape[0], Bi.shape[1]))
        # Bi----k*R
        for j in range(Bi.shape[0]):
            for k in range(Bi.shape[1]):
                r = [-1, 1]
                i = random.randint(0, 1)
                eps = r[i]
                eps_Bi[j, k] = eps
                hx_Bi[j, k] = eps * Bi[j, k]
        return hx_Bi, eps_Bi, Bi
    def cal_dis1(self, data_enc_point):
        Bi = data_enc_point - self.centers
        hx_Bi = np.empty((Bi.shape[0], Bi.shape[1]), dtype=EncryptedNumber)
        eps_Bi = np.empty((Bi.shape[0], Bi.shape[1]))
        # Bi----k*R
        for j in range(Bi.shape[0]):
            for k in range(Bi.shape[1]):
                r = [-1, 1]
                i = random.randint(0, 1)
                eps = r[i]
                eps_Bi[j, k] = eps
                hx_Bi[j, k] = eps * Bi[j, k]
        return hx_Bi, eps_Bi, Bi

    def cal_dis2(self, iter_data):
        res = np.empty((iter_data[0].shape[0], iter_data[0].shape[1]))
        d = []
        # dis_start = time.perf_counter()
        for j in range(iter_data[0].shape[0]):
            dj = self.pubkey.encrypt(0)
            for k in range(iter_data[0].shape[1]):
                if iter_data[1][j, k] == 1:
                    if iter_data[0][j, k] <= 0:
                        res[j, k] = -1
                    elif iter_data[0][j, k] == 1:
                        res[j, k] = 1
                else:
                    if iter_data[0][j, k] <= 0:
                        res[j, k] = 1
                    elif iter_data[0][j, k] == 1:
                        res[j, k] = -1
                dj += iter_data[2][j, k] * res[j, k]

            d.append(dj)
        return d
    def cal_dis(self, data_enc_point):
        server = self.server
        pool = Pool(cpu_number)
        hx_Bi_eps_Bi_Bi = []
        cal_dis1_start=time.perf_counter()
        hx_Bi_eps_Bi_Bi.append(pool.map(self.cal_dis1, data_enc_point))
        pool.close()
        pool.join()
        hx_Bi = []
        eps_Bi = []
        Bi = []
        for i in hx_Bi_eps_Bi_Bi[0]:
            hx_Bi.append(i[0])
            eps_Bi.append(i[1])
            Bi.append(i[2])
        print("cal_dis,ciphertext_calculation_time:%f"%(time.perf_counter()-cal_dis1_start))

        # 将混淆的BI传输到S2
        tx_start=time.perf_counter()
        print('cal_dis,send/receive size:')
        send(server, hx_Bi)
        hx_Bi_sig = receive(server)
        print('cal_dis,socket_time:%f'%(time.perf_counter()-tx_start))

        cal_dis2_start=time.perf_counter()
        for i in range(len(hx_Bi_eps_Bi_Bi[0])):
            hx_Bi_eps_Bi_Bi[0][i] = list(hx_Bi_eps_Bi_Bi[0][i])

        for i in range(len(hx_Bi_eps_Bi_Bi[0])):
            hx_Bi_eps_Bi_Bi[0][i][0] = hx_Bi_sig[0][i]
        d_batch=[]
        pool=Pool(cpu_number)
        d_batch.append(pool.map(self.cal_dis2, hx_Bi_eps_Bi_Bi[0]))
        pool.close()
        # pool.terminate()
        pool.join()
        print("cal_dis,ciphertext_sign_time:%f"%(time.perf_counter() - cal_dis2_start))
        return d_batch[0]

    def compare(self, e1, e2):
        server = self.server
        ec = e1 - e2
        r = [-1, 1]
        i = random.randint(0, 1)
        eps = r[i]
        ec1 = eps * ec
        send(server, ec1)
        sig = receive(server)
        if sig==-1:
            res=e1
        elif sig==0:
            if eps==1:
                res=e1
            else:
                res=e2
        elif sig==1:
            if eps==1:
                res=e2
            else:
                res=e1

        return res
    def find_min_dis(self, d):
        min_d = d[0]
        for i in range(1,len(d)):
            min_d=self.compare(min_d,d[i])
        return min_d
    def get_one(self,one):
        sub_dis = []
        sub_eps = []
        list_r = [-1, 1]
        for j in range(K):
            sub_dis.append(np.array(one[j]) - np.array(one))
            sub_eps.append([random.choice(list_r) for i in range(K)])
        return sub_dis,sub_eps

    def find_min_dis_batch(self, d):
        d = np.array(d)
        min_col = d[:,0]
        
        for cidx in range(1, K):
            start = time.perf_counter()
            current_col = d[:, cidx]
            diff_col = current_col - min_col
            socket_start = time.perf_counter()
            print('find_min_dist,l=%d,send/receive size:'%cidx)
            send(self.server, diff_col)
            diff_sign = np.array(receive(self.server))
            print('find_min_dist,l=%d,socket_time:%f'%(cidx,time.perf_counter()-socket_start))
            min_col[diff_sign<=0] = current_col[diff_sign<=0]
            print('find_min_dist,l=%d,time:%f'%(cidx,time.perf_counter()-start))

        return min_col


    def cfp(self, data):
        
        for ind in range(0,len(data),batch_size):
            print("batch index=%d"%(ind))

            dbatch = self.cal_dis(data[ind:ind+batch_size])
            min_d = self.find_min_dis_batch(dbatch)

            stage3_start=time.perf_counter()
            for i,d in enumerate(min_d):
                self.cluster[dbatch[i].index(d)].append(i+ind)
            print("reassign time:%f"%(time.perf_counter()-stage3_start))

    
    def updata_enc_center(self, enc_data):
        start=time.perf_counter()
        for label, inds in self.cluster.items():
            if len(enc_data[inds]) !=0:
                self.centers[label] = np.mean(enc_data[inds], axis=0)
        print('update_enc_center time:%f'%(time.perf_counter()-start))
    
    def enc_divide(self,data):
        tmp_cluster = copy.deepcopy(self.cluster)
        for label, inds in tmp_cluster.items():
            data_label=data[inds]
            for ind in range(0, len(data_label), batch_size):
                print("batch index=%d"%(ind))
                dbatch = self.cal_dis(data_label[ind:ind + batch_size])
                min_d = self.find_min_dis_batch(dbatch)

                reassign_start = time.perf_counter()
                for i, d in enumerate(min_d):
                    new_label = dbatch[i].index(d)
                    if new_label == label:  # 若类标记不变，跳过
                        continue
                    else:
                        self.cluster[label].remove(inds[i+ind])
                        self.cluster[new_label].append(inds[i+ind])
                print("reassign time:%f"%(time.perf_counter()-reassign_start))

    def enc_fit(self, data):
        stage1_start=time.perf_counter()
        data_enc = self.initial_1(data)
        print("initialization time:%f"%(time.perf_counter()-stage1_start))
        stage2_start = time.perf_counter()
        self.cfp(data_enc)
        print("cfp time:%f"%(time.perf_counter() - stage2_start))
        step = 0
        while step < self.maxstep:


            step += 1
            print("step=%d"%step)
            self.updata_enc_center(data_enc)
            self.enc_divide(data_enc)
        self.server.close()


def test_enc(n):
    km = KMEANS(n)
    start=time.perf_counter()
    km.enc_fit(data)
    end=time.perf_counter()-start
    print(end)
    prec_label = []
    for lb, inds in km.cluster.items():
        for i in inds:
            prec_label.append([i, lb])
    dic = {}
    for number in prec_label:
        key = number[0]
        dic[key] = number[1]
    s = sorted(dic.items(), key=lambda x: x[0])
    y_pred = []
    for si in s:
        y_pred.append(si[1])
    print(y_pred)
    with open(config._DATA_PATH+"_%d_pred"%K, "w") as pred_out:
        pred_out.write(str(y_pred))
    print("end")


if __name__ == '__main__':
    fdata = load_data()
    data = np.array(fdata)
    data[data<config._EQUAL_TOL] = 0.
    
    
    for k in config._K:
        K = k
        start=time.perf_counter()
        test_enc(k)
        cost_t = time.perf_counter()-start 
        with open(config._DATA_PATH+"_%d_time"%k, 'w') as t_out:
            t_out.write(str(cost_t))    
