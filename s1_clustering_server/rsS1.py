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
from sklearn.metrics import normalized_mutual_info_score as nmi 
from sklearn.metrics import adjusted_rand_score as ari

private_key = None

def log(content):
    print('[%s] %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))), content))

def load_data(data_path):
    points = np.loadtxt(data_path)
    return points

def init_encryption(dec_server_ip, dec_server_port):
    '''
        Initialization of encryption:
            1. connect to decryption server
            2. deliver the private key
    '''
    global private_key
    start_ts = time.perf_counter()
    public_key, private_key = paillier.generate_paillier_keypair()        
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((dec_server_ip, dec_server_port))
    send(server, {'type': 'key',  'content':private_key} )
    resp = receive(server)
    print('Time of delivering private key: %d, response: %s'%(time.perf_counter() - start_ts, resp['content']))
    return server, public_key

def pool_minus(x):
    return x[0] - x[1]

def pool_mul_sum(x):
    m = x[0] * x[1] * x[2]
    return np.sum(m, axis=len(m.shape)-1)

class CryptoRandomSwap:

    def __init__(self, enc_X, n_cluster, T, km_T, server, public_key, batch = config._BATCH_SIZE):
        '''
            init function of CryptoRandonSwap

            argv:
                @X:
                    np.ndarray, the data matrix
                @n_cluster: 
                    int, the number of cluster
                @T:
                    int, the number of iterations
                @km_T:
                    int, the number of iterations in KMeans
                @server:
                    the decryption server
                @public_key:
                    the public key
                @batch:
                    int, the size of the batch
        '''
        
        # RandomSwap related variable
        ## the encrypted data
        self.enc_X = enc_X
        ## size of the data
        self.N = self.enc_X.shape[0]
        ## dimension of the data
        self.D = self.enc_X.shape[1]
        ## the number of the target clusters
        self.n_cluster = n_cluster
        ## the number of the random swap iterations
        self.T = T
        ## the number of the kmeans' iterations
        self.km_T = km_T 
        ## the centroids
        self.C = np.empty((self.n_cluster, self.D), dtype=paillier.EncryptedNumber)
        ## the label of each data point
        self.P = np.array([0] * self.N)
        ## distance betw each data point and their nearest centroid
        self.X_C_dist = np.empty((self.N, ), dtype=EncryptedNumber)
        ## the size of the batch
        self.batch = batch 
        
        # Encryption related variabels
        self.server = server
        self.public_key = public_key
        
        # loss
        self.loss = None

    def _sign(self, data):
        '''
            Judge the sign of numbers in data

            argv:
                @data:
                    int / np.ndarray
            
            return:
                int -> int 
                np.ndarray -> np.ndarray

                1  ==> >0
                -1 ==> <0
                0  ==> ==0
        '''

        if type(data) is paillier.EncryptedNumber:
            t = 'sign'
        elif type(data) is np.ndarray:
            t = 'sign_batch'
        else:
            return None
        # log('_sign data size: %s'%str(data.__sizeof__()))
        req = {'type':t, 'content': data}
        # log('_sign req size: %s'%str(req.__sizeof__()))
        send(self.server, req)
        resp = receive(self.server)
        return resp['content']
    
    def _select_random_obj(self, m):
        '''
            select random data object from self.enc_X

            argv:
                @m:
                    int, the selected one should not be one of self.C[0:m]
            
            return:
                int, the index of the selected object
        '''
        while True:
            i = random.randrange(0, self.N)
            flag = True 

            for j in range(0, min(m, self.n_cluster)):
                signs = self._sign(self.enc_X[i]-self.C[j])
                if True == (signs==0).all():
                    flag = False 
                    break 
            if flag:
                break
        
        return i

    def _select_random_rep(self):
        '''
            select self.n_cluster centroids without duplications
        '''

        for i in range(self.n_cluster):
            self.C[i] = self._select_random_obj(i)

    def _swap(self):
        '''
            swap a centroid

            return:
                int, the index of the swapped centroid
        '''
        
        j = random.randrange(0, self.n_cluster)
        self.C[j] = self._select_random_obj(self.n_cluster)

        return j

    def _row_min(self, mat):
        '''
            find the minimal val of each row

            argv:
                @mat:
                    np.ndarray, a matrix
        '''
        min_col = mat[:, 0]
        min_col_idx = np.array([0] * mat.shape[0])

        for cidx in range(1, mat.shape[1]):
            current_col = mat[:, cidx]
            diff_col = current_col - min_col
            diff_sign = self._sign(diff_col)
            min_col[diff_sign<0] = current_col[diff_sign<=0]
            min_col_idx[diff_sign<0] = cidx
        
        return min_col, min_col_idx

    def _l1_dist(self, data_batch_1, data_batch_2, all_pair=False):
        '''
            calculate the L1 distance between elements in @data_batch_1
            and @data_batch_2, e.g.:
                
                @data_batch_1 = [v_1, v_2, ..., v_batch]
                @data_batch_2 = [z_1, z_2, ..., z_batch]

                return:
                if all_pair == False:
                    [
                        l1_dist(v_1, z_1),
                        l1_dist(v_2, z_2),
                        ...
                        l1_dist(v_batch, z_batch)
                    ]
                else:
                    [
                        [ l1_dist(v_1, z_1), l1_dist(v_1, z_2), ..., l1_dist(v_1, z_{batch_2}) ],
                        ...
                        [ l1_dist(v_{batch_1}, z_1), l1_dist(v_{batch_1}, z_2), ..., l1_dist(v_{batch_1}, z_{batch_2}) ]
                    ]
 
            argv:
                @data_batch_1:
                    np.ndarray, shape=(batch, self.D)
                @data_batch_2:
                    np.ndarray, shape=(batch, self.D)
                @all_pair:
                    True: calcualte all pairs
                    False: only calculate corresponding pairs

            return:
                np.ndarray, shape=(len(data_batch_1), len(data_batch_2)) if all_pair=True;
                else shape=(len(data_batch_1), )
        '''
        if 1 == len(data_batch_1.shape):
            data_batch_1 = np.array([data_batch_1])
        if 1 == len(data_batch_2.shape):
            data_batch_2 = np.array([data_batch_2])

        pool = Pool(config._S1_CPU_NUMBER)

        if all_pair == False:
            diff = data_batch_1 - data_batch_2
        else:

            diff_x_y = [[v, data_batch_2] for v in data_batch_1]
            diff = np.array(pool.map(pool_minus, diff_x_y))

        eps = np.random.choice((-1,1), size=diff.shape)
        confused_diff = diff * eps
        log('l1 distance: sign: start')
        confused_sign = self._sign(confused_diff)
        log('l1 distance: sign: end')
        dist_mat = np.array(pool.map(pool_mul_sum, zip(diff, confused_sign, eps)))
        log('l1 distance: dist_mat: done')

        pool.close()
        pool.join()

        return dist_mat
        
    def _local_repartition(self, j):
        '''
            update the partition after random swap

            argv:
                @j:
                    int, the index of the swapped centroid
        '''
        
        # object rejection
        old_idx = np.where(self.P==j)[0]
        for i in range(0, len(old_idx), self.batch):
            batch_idx = old_idx[i: i+self.batch]
            dist_mat = self._l1_dist(self.enc_X[batch_idx], self.C, all_pair=True)
            nearest_dist, nearest_label = self._row_min(dist_mat)
            self.P[batch_idx] = nearest_label
            self.X_C_dist[batch_idx] = nearest_dist
        
        # object attraction
        for i in range(0, self.N, self.batch):
            tail = i+self.batch if i+self.batch<=self.N else self.N
            batch_idx = range(i, tail)
            
            new_dist = self._l1_dist(self.enc_X[batch_idx], self.C[j], all_pair=True)
            old_dist = self.X_C_dist[batch_idx].reshape((len(batch_idx), 1))
            nearest_dist, nearest_dist_label = self._row_min(np.concatenate((new_dist, old_dist), axis=1))
            self.P[i+np.where(nearest_dist_label==0)[0]] = j
            self.X_C_dist[batch_idx] = nearest_dist

    def _optimal_partition(self):
        '''
            update the self.P and self.X_C_dist based on self.enc_X and self.C
        '''

        for i in range(0, self.N, self.batch):
            tail = i+self.batch if i+self.batch<=self.N else self.N
            batch_idx = range(i, tail)
            log('optimal partition: l1_dist: %d'%i)
            dist_mat = self._l1_dist(self.enc_X[batch_idx], self.C, all_pair=True)
            log('optimal partition: row_min: %d'%i)
            nearest_dist, nearest_label = self._row_min(dist_mat)
            log('optimal partition: row_min end: %d'%i)
            self.P[batch_idx] = nearest_label
            self.X_C_dist[batch_idx] = nearest_dist

    def _optimal_representative(self):
        '''
            update the self.C based on the self.P
        '''
        new_C = np.empty((self.n_cluster, self.D), dtype=paillier.EncryptedNumber)
        new_C[:,:] = self.public_key.encrypt(0)

        cnt = np.array([0] * self.n_cluster)
        for i in range(self.N):
            label = self.P[i]
            new_C[label] += self.enc_X[i]
            cnt[label] += 1
        
        for c in range(self.n_cluster):
            if cnt[c] != 0:
                self.C[c] = new_C[c] / cnt[c]
        
    def _km(self):
        '''
            run kmeans for self.km_T iterations
        '''
        for t in range(self.km_T):
            log('  km_T: %d'%t)
            self._optimal_representative()
            self._optimal_partition()

    def _mean_absolute_error(self):
        '''
            calculate the mean absolute error using self.X_C_dist
        '''

        return np.sum(self.X_C_dist)

    def fit(self):

        # self._select_random_rep()
        ctr_idx = np.random.choice(range(0, self.N), self.n_cluster, replace=False)
        self.C = self.enc_X[ctr_idx]
        log('optimal partition')
        self._optimal_partition()
        self.loss = self._mean_absolute_error()

        for t in range(self.T):
            log('rs_T: %d'%t) 
            
            # cache the old state
            old_C = np.copy(self.C)
            old_P = np.copy(self.P)
            old_X_C_dist = np.copy(self.X_C_dist)
            old_loss = self.loss


            # process the clustering
            log('swap')
            j = self._swap()
            log('local repartition')
            self._local_repartition(j)
            log(self.P)
            self._km()
            self.loss = self._mean_absolute_error()

            # whether a better result?
            log(str(private_key.decrypt(self.loss)))
            log(str(private_key.decrypt(old_loss)))
            if self._sign(self.loss - old_loss) > 0: ## not a better result
                log('not  better')
                self.C = old_C
                self.P = old_P
                self.X_C_dist = old_X_C_dist
                self.loss = old_loss
        
        return self.C, self.P


log('Initialization')
server, public_key = init_encryption(config._S2_IP, config._S2_PORT)
X = load_data(config._DATA_PATH)
# truth = label = [ 1 ] * 1000 + [ 2 ] * 600 + [ 3 ] * 600 + [ 4 ] * 800
X = np.concatenate((X[0:100], X[1000:1060, :], X[1600:1660, :], X[2200:2280, :]), axis=0)
truth = label = [ 1 ] * 100 + [ 2 ] * 60 + [ 3 ] * 60 + [ 4 ] * 80

log('Encrypt data')
shape = X.shape
flatten_X = X.flatten()
pool = Pool(config._S1_CPU_NUMBER)
flatten_enc_X = pool.map(public_key.encrypt, flatten_X)
pool.close()
pool.join()

enc_X = np.array(flatten_enc_X).reshape(shape)


log('Clustering')
for k in config._K:
    rs = CryptoRandomSwap(enc_X, k, 20, 2, server, public_key)
    C, P = rs.fit()
    n = nmi(truth, P)
    a = ari(truth, P)
    log('k=%d, nmi=%f, ari=%f'%(k, n, a))



