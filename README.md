# encryption_clustering - Multiple thread version
Implementation of CTKM.

The code is implemented by Shuhai Meng and modified and archived by Yinjia Zhang.

## Framework

There are two servers: s1/s2。

- s1: clustering server
- s2: decryption server

The environments of the two servers are the same and as following:

- python3
- python phe库
- python gmpy2库

Developing code structure:
```
.
├── data                   // data dir
│   └── {datafiles}
├── local_dev              // local development bash and configuration files
│   ├── config.py
│   └── dev.sh
├── README.md
├── remote_dev             // remote development bash and configuration files
│   ├── config.py
│   └── dev.sh
├── s1_clustering_server   // code of s1 server
│   ├── help.py
│   └── kS1.py
└── s2_decryption_server   // code of s2 server
    ├── help.py
    └── kS2.py

```

s1 server dir:
```
.
├── clustering_server         // code dir
│   ├── config.py             // configuration files, uploaded by remote_dev/dev.sh, the same with that in s2 server
│   ├── help.py
│   └── kS1.py
└── data                      // data files
    └── yelp3000_process.txt

```

s2 server  dir:
```
.
└── decryption_server         // code dir
    ├── config.py             // configuration files, uploaded by remote_dev/dev.sh, the same with that in s1 server
    ├── help.py
    └── kS2.py
```


## Development

run the dev.sh bash to upload the code and data to the two servers. The two servers should be able to communicate with
each other.

## Run

### 1. On s2 server:
```python
python3 {dir_path}/kS2.py 
```
### 2. On s1 server
```
python3 {dir_path}/kS1.py
```

## Notice

### 1.  When you change the IPs of s1 or s2, remember to change those in both config.py asn dev.sh


