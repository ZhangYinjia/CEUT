#coding:utf-8

import sys
sys.path.append(sys.path[0] + '/../')
from config.load_config import Config
import csv
import json
import time

class DataLoader:

    def __init__(self, data_file_name=None, cate_file_name=None, business_file_name=None):
        '''
            load config and category information from processed data
        '''

        #load config
        config = Config().config
        self.data_file_name = config['data_file_name'] if data_file_name is None else data_file_name
        self.cate_file_name = config['cate_file_name'] if cate_file_name is None else cate_file_name
        self.business_file_name = config['business_file_name'] if business_file_name is None else business_file_name

        #load category
        self.cate_parent = {}
        with open(self.cate_file_name) as cate_csv:
            try:
                cate_rec = csv.reader(cate_csv)
                for row in cate_rec:
                    self.cate_parent[row[0]] = row[1]
            except Exception as e:
                raise e
        
        #load business's category info
        with open(self.business_file_name) as businss_cate_f:
            if 'line' in self.business_file_name:
                self.business_cate = {}
                line = businss_cate_f.readline()
                while line!='' and line!='\n':
                    item = json.loads(line)
                    self.business_cate[item['bid']] = item['category']
                    line = businss_cate_f.readline()
            else:
                self.business_cate = json.load(businss_cate_f)

    def get_cate_list(self, cate):
        '''
            get a category's category path from root category

            @cate: the most detailed category

            #return: [root, cate_1, cate_2, ..., cate]
        '''
        path = []
        p = cate
        while p is not None and p != '':
            path.insert(0,p)
            p = self.cate_parent[p]
        return path

    def get_business_cate_path(self, bs_name):
        '''
            get all category path of a business

            @bs_name: business name to get all category paths

            #return: [ [<path_1>], [<path_2>], ...]
        '''
        paths = []
        if 'line' in self.business_file_name:
            return  self.business_cate[bs_name]
        else:
            tail_cate = self.business_cate[bs_name]
            for tc in tail_cate:
                path = self.get_cate_list(tc)
                paths.append(path)
        return paths
    
    def get_all_cate_path(self):
        '''
            get all category paths

            #return [ [<path_1>], [<path_2>], ...]
        '''
        paths = []
        for bs in self.business_cate.keys():
            paths += self.get_business_cate_path(bs)
        return paths

    def load(self, convert_func, **kwargs):
        '''
            load a user's all category path
        
            @convert_func: convert function, to convert 
            category path to data point. Parameters of convert_func
            is: uid, business_categorypath_dict and a kwargs. uid is
            the id of a user, business_categorypath_dict
            is a dict {business_1: [[path_1],[path_2],...], ...} and
            kwargs is a list of other helpful parameters.
            Return of the fucntion is a data node
            @kwargs: a dict of other parameters to deliver to convert_func 
            and some other parameters

            #return: a list of data node
        '''
        if type(convert_func).__name__ != 'function':
            raise Exception('convert_func must be a function')
        print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' begin loading...')
        user_data = {}

        with open(self.data_file_name) as user_data_f:
            if 'line' in self.data_file_name:
                line = user_data_f.readline()
                while line!='' and line!='\n':
                    item = json.loads(line)
                    user_data[item['uid']] = item['business']
                    line = user_data_f.readline()
            else:
                user_data = json.load(user_data_f)
        
        
        valid_uid = kwargs['valid_uid'] if 'valid_uid' in kwargs else user_data.keys()
        data_size = kwargs['data_size'] if 'data_size' in kwargs else float('inf')
        ret_data = [None for i in range(len(user_data.keys()))] if valid_uid is None else [None for i in range(len(valid_uid))]

        print ("size of valid users: %d"%len(valid_uid))

        for count, uid in enumerate(valid_uid):
            if count+1 > data_size:
                break

            if count % 100==0:
                print (count)
            bus_dict = {}
            for bid in user_data[uid]:
                if bid in self.business_cate:
                    bus_dict[bid] = self.get_business_cate_path(bid) if 'line' not in self.business_file_name else self.business_cate[bid]
            index = count if len(valid_uid)==len(user_data) else valid_uid.index(uid)
            ret_data[index] = convert_func(uid, bus_dict, kwargs)
        print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' loading finished')
        return ret_data

        
