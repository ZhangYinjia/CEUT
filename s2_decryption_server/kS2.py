import socket
from help import send,receive
import numpy as np
from multiprocessing import Pool
import time
import math
import config

'''
Modified Method:
    hx_Bi_one
'''


class KS2:
    def __init__(self,):
        self.secret_key=None

    def initial(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (config._S2_IP, config._S2_PORT)
        print("Starting up on %s:%s" % server_address)
        sock.bind(server_address)
        sock.listen(10)
        return sock

    def hx_Bi_one(self,hx_Bi):
        
        hx_Bi_sig = np.zeros((hx_Bi.shape[0], hx_Bi.shape[1]))
        for j in range(hx_Bi.shape[0]):
            hx_Bi_sig[j, :] = [self.secret_key.decrypt(e) for e in hx_Bi[j, :]]
        
        hx_Bi_sig[hx_Bi_sig > 0] = 1
        hx_Bi_sig[hx_Bi_sig <= 0] = 0
        return hx_Bi_sig
       
    def dec(self,x):
        
        e = config._EQUAL_TOL
        plain_text = self.secret_key.decrypt(x)
        if plain_text < -e:
            sig = -1
        elif plain_text > e:
            sig = 1
        else:
            sig = 0
        return sig

if __name__ == '__main__':
    cpu_number=config._S2_CPU_NUMBER
    ks2 = KS2()
    e=config._EQUAL_TOL
    sock=ks2.initial()
    while True:
        for k in config._K:
            print("K=%d, waiting for a connection"%k)
            connection, client_address = sock.accept()
            try:
                number = config._NUMBER
                
                batch_size = config._BATCH_SIZE

                max_step = config._MAX_STEP
                print("Connection from", client_address)
                ks2.secret_key = receive(connection)
                print("recieved secret key!")
                pool = Pool(cpu_number)
                for ite in range(max_step+1):
                    print('step=%d'%ite)
                    step=0
                    tatol_size=0
                    while(step < math.ceil(number / batch_size)):

                        print('batch index=%d'%step)
                        hx_Bi = receive(connection)

                        hx_Bi_sig = []
                        hx_Bi_start=time.perf_counter()
                        hx_Bi_sig.append(pool.map(ks2.hx_Bi_one, hx_Bi))
                        print("cal_dist_s2_time:%f"%(time.perf_counter()-hx_Bi_start))

                        send(connection, hx_Bi_sig)
                        size=len(hx_Bi)
                        tatol_size+=size
                        if size % batch_size!=0:
                            step-=1
                        
                        
                        for l in range(1, k):
                            hx_dis=receive(connection)
                            start = time.perf_counter()
                            hx_dis_np=np.array(hx_dis).flatten()
                            hx_dis_np_sig=pool.map(ks2.dec,hx_dis_np)
                            hx_dis_np_sig_np=np.array(hx_dis_np_sig).reshape(np.array(hx_dis).shape)
                            print('find_min_dist,l=%d,time:%f'%(l,time.perf_counter()-start))
                            send(connection, hx_dis_np_sig_np)
                        
                        step+=1
                        if tatol_size==number:
                            print("一次迭代完成")
                            break
                pool.close()
                pool.join()            
            finally:
                connection.close()
