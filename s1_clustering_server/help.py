import pickle
def receive(sock):
    rawsize = sock.recv(1024)
    size = int.from_bytes(rawsize, 'big')
    r = buf_r = b''
    while len(r) < size:
        buf_r = sock.recv(size - len(r))
        if not buf_r:
            if len(r) < size: print("###Predicting pickle error")
            break
        r += buf_r
    # print("received bytes:%d"%len(r))
    return pickle.loads(r)

def send(sock, data):
    # print('send: data size: %s'%str(data.__sizeof__()))
    pdata = pickle.dumps(data)
    # print('send: pdata size: %s'%str(pdata.__sizeof__()))
    # print('send: len of pdata: %s'%str(len(pdata)))
    size = len(pdata)
    rawsize = size.to_bytes(1024, byteorder='big')
    # print("sent bytes:%d"%size)
    sock.sendall(rawsize)
    sock.sendall(pdata)