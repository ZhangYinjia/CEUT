import pickle
def receive(sock):
    rawsize = sock.recv(1024)
    size = int.from_bytes(rawsize, 'big')
    print(size)

    r = buf_r = b''
    while len(r) < size:

        buf_r = sock.recv(size - len(r))
        if not buf_r:
            if len(r) < size: print("###Predicting pickle error")
            break
        r += buf_r
    return pickle.loads(r)

def send(sock, data):
    pdata = pickle.dumps(data)
    size = len(pdata)
    rawsize = size.to_bytes(1024, byteorder='big')
    sock.sendall(rawsize)
    sock.sendall(pdata)