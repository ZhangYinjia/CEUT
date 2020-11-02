import socket
from help import send,receive
import numpy as np
from multiprocessing import Pool
import time
import math
import config


class DecryptionServer:

    def __init__(self, ip, port, zero_tol):
        '''
            Initialization the decryption server
        '''
        self.ip = ip
        self.port = port 
        self.zero_tol = zero_tol
        self.private_key = None 
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.ip, self.port)
        print("Decryption server starting on %s:%s" % server_address)
        self.sock.bind(server_address)
        self.sock.listen(10)

    def _assign_key(self, req):
        '''
            assign the private key
        '''
        if self.private_key is None:
            self.private_key = req['content']
            return {'type': 'text', 'content': 'OK'}
        else:
            return {'type': 'text', 'content': 'REFUSED'}

    def _sign(self, req):
        '''
            return the sign of the number in req
        '''
        
        number = req['content']
        plaintext = self.private_key.decrypt(number)
        response = {'type': 'int'}
        if plaintext > self.zero_tol:
            response['content'] = 1
        elif plaintext < - self.zero_tol:
            response['content'] = -1
        else:
            response['content'] = 0
        
        return response
    
    def _sign_batch(self, req):
        '''
            return signs of the numbers in req
        '''
        arr = np.array(req['content'])
        arr_shape = arr.shape
        arr = arr.flatten()

        pool = Pool(config._S2_CPU_NUMBER)
        signs = np.array(pool.map(self.private_key.decrypt, arr))
        pool.close()
        pool.join()
        signs = signs.reshape(arr_shape)
        
        signs[signs > self.zero_tol] = 1
        signs[signs < - self.zero_tol] = -1

        return  {'type': 'array', 'content': signs}
        

    def run(self):
        '''
            Run the decryption server, there are three kinds of requests:
                1. {'type':'key', 'content': PaillierPrivateKey, 'text': str}
                2. {'type':'sign', 'content': float, 'text': str}
                3. {'type':'sign_batch', 'content': np.ndarray, 'text':str}
        '''
        connection, client_address = self.sock.accept()
        while True:
            request = receive(connection)
            req_type = request['type']
            print("[%s] %s"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))), request['type']))
            if 'key'==req_type:
                response = self._assign_key(request)
            elif 'sign'==req_type:
                response =  self._sign(request)
            elif 'sign_batch'==req_type:
                response = self._sign_batch(request)
            print("[%s] %s"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))), 'process done'))
            send(connection, response)

server = DecryptionServer(config._S2_IP, config._S2_PORT, config._EQUAL_TOL)
server.run()





