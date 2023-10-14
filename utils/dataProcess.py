from audioop import bias
import scipy.io as scio
import numpy as np
import dgl, torch
import torch
import torch.nn as nn
import scipy.sparse as sp
from tqdm import tqdm

# 读取.mat数据
def get_data_from_matfile(dataset):

    rate_path = "./data/{}/rating.mat".format(dataset)
    trust_path = "./data/{}/trustnetwork.mat".format(dataset)

    rating_data=scio.loadmat(rate_path)['rating'].astype(np.int32)
    trust_data=scio.loadmat(trust_path)['trustnetwork'].astype(np.int32)

    if dataset == "epinions":
        rating_data = rating_data[:,[0,1,3]]
    elif dataset == "ciao":
        rating_data = rating_data[:,[0,1,3]]
    else:
        print("no such dataset")

    return rating_data, trust_data

def load_data_yelp(path = '/home/lgq/project/myMCL/data/yelp/'):

    # origin / unfiltered data
    rating_data = np.load(path + 'rating_data.npy')
    trust_data = np.load( path + 'trust_data.npy')

    user_dict = {}
    item_dict = {}

    for a,b,r in rating_data:
            user_dict[a] = user_dict.get(a,0) + 1
            item_dict[b] = item_dict.get(b,0) + 1

    # filter user and item
    user_set = set(u for u,c in user_dict.items() if c >= 20)
    item_set = set(u for u,c in item_dict.items() if c >= 10)


    rating_data_filtered = []
    for a,b,r in rating_data:
        if a in user_set and b in item_set :
            user_set.add(a)
            item_set.add(b)
            rating_data_filtered.append([a,b,r])

    trust_data_filtered = []
    for a,b, in trust_data:
        if a in user_set and b in user_set :
            trust_data_filtered.append([a,b])


    len(rating_data_filtered), len(trust_data_filtered), len(user_set), len(item_set)

    # adjust index of user/item
    rating_data_final, trust_data_final = [], []
    user_index, item_index = {}, {}
    for a,b,r in rating_data_filtered:
        if a not in user_index:
            user_index[a] = len(user_index)
        a_index = user_index[a]

        if b not in item_index:
            item_index[b] = len(item_index)
        b_index = item_index[b]

        rating_data_final.append([a_index, b_index, r])

    for a,b in trust_data_filtered:
        trust_data_final.append([user_index[a],user_index[b]])

    return np.array(rating_data_final), np.array(trust_data_final)


def load_data_1m(path='data/MovieLens_1M/', delimiter='::', frac=0.1, seed=1234):


    # print('reading data...')
    data = np.loadtxt(path+'movielens_1m_dataset.dat', skiprows=0, delimiter=delimiter).astype('int32')

    return data[:,[0,1,2]]

def load_data_filmtrust():
    path = 'data/filmtrust/'
    rating_file = open(path + 'ratings.txt')
    trust_file = open(path + 'trust.txt')

    rating_data, trust_data = [],[]
    for line in rating_file:
        rating_data.append(list(map(float,line.split())))
    for line in trust_file:
        trust_data.append(list(map(float,line.split())))
    rating_data = np.array(rating_data)
    trust_data = np.array(trust_data)

    return rating_data, trust_data
    







def load_data_100k(path='data/100k/', delimiter='\t'):

    train = np.loadtxt(path+'movielens_100k_u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'movielens_100k_u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        test_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m
    

# def data_clean(rating_data, trust_data, remove_no_trust_user=False):
#     # remove duplicate rate, each user-item rating only keep one record.
#     temp_set = set()
#     for line in rating_data:
#         key =  (line[0],line[1])
#         if key in temp_set:
#             line[2] = -1
#         else:
#             temp_set.add(key)     
#     rating_data = rating_data[rating_data[:,2] != -1]

#     if remove_no_trust_user:
#         u_set  = set(trust_data.reshape(-1))
#         u2o, i2o = {}, {}
#         i = 0
#         for u in u_set:
#             if u not in u2o:
#                 u2o[u] = i
#                 i += 1
#         n = 0
#         for line in rating_data:
#             if line[0] not in u_set:
#                 line[2] = -1
#                 n += 1
#         rating_data = rating_data[rating_data[:,2] != -1]

#         j = 0
#         for i in set(rating_data[:,1]):
#             if i not in i2o:
#                 i2o[i] = j
#                 j += 1

#         for line in trust_data:
#             line[0] = u2o[ line[0]]
#             line[1] = u2o[ line[1]]

#         for line in rating_data:
#             line[0] = u2o[ line[0]]
#             line[1] = i2o[ line[1]]
#         user_num, item_num = len(u2o), len(i2o)
#     else:
#         # make idx start from 0
#         if min(rating_data[:,0]) == 1:   # if user-id strat from 1
#             rating_data[:,0] -= 1
#             trust_data[:,0] -= 1
#             trust_data[:,1] -= 1
#         if min(rating_data[:,1]) == 1:   # if item-id start from 1
#             rating_data[:,1] -= 1
#         item_num = max(rating_data[:,1]) + 1
#         rating_user_num = max(rating_data[:,0]) + 1
#         trust_user_num = max(trust_data.reshape(-1)) + 1
#         user_num = max(rating_user_num, trust_user_num)
    
#     print("Data info: user_num: {}, item_num {}, trust_num: {}, rating record num: {} "
#                     .format(user_num, item_num, len(trust_data), len(rating_data) ))
    
#     return rating_data, trust_data, user_num, item_num
def data_clean(rating_data, trust_data = None):
    ''''''
    # 去除重复记录
    temp_set = set()
    for line in rating_data:
        key =  (line[0],line[1])
        if key in temp_set:
            line[2] = -1
        else:
            temp_set.add(key)     
    rating_data = rating_data[rating_data[:,2] != -1]

    user_set = np.unique(rating_data[:,0])
    item_set = np.unique(rating_data[:,1])
    if trust_data is not None:
        user_set = set(user_set).union(np.unique(trust_data[:,[0,1]]))
    user_num = len(user_set) # num of users
    item_num = len(item_set)  # num of movies

    user_dict = {}
    for i, u in enumerate(user_set):
        user_dict[u] = i
    item_dict = {}
    for i, m in enumerate(item_set):
        item_dict[m] = i

    # re-order
    for line in rating_data:
        line[0] = user_dict[line[0]]
        line[1] = item_dict[line[1]]
    
    # print("Data info: user_num: {}, item_num {}, rating record num: {} "
    #                 .format(user_num, item_num, len(rating_data) ))

    if trust_data is not None:
        for line in trust_data:
             line[0] = user_dict[line[0]]
             line[1] = user_dict[line[1]]
        # print("trust_num:", len(trust_data))
        trust_data = trust_data[trust_data[:,0] != trust_data[:,1]]
   
    return rating_data, trust_data, user_num, item_num

def data_split(rating_data, train_ratio=0.8):
    # 划分训练集，测试集，验证机  a:b:b   a+b+b=1
    np.random.seed(36)
    np.random.shuffle(rating_data)
    rating_data_len = len(rating_data)

    train_size = int(rating_data_len * train_ratio)
    valid_size = train_size + (rating_data_len - train_size)//2

    rating_data_train = rating_data[:train_size]
    rating_data_valid = rating_data[train_size:valid_size]
    rating_data_test  = rating_data[valid_size:]
    return rating_data_train, rating_data_valid, rating_data_test
    

def add_pref_col(rating_data, user_info, item_info):
    '''user_dict[user_id] = [ user_info["record_num"]  , user_info["avg_rate"],  user_info["norm
    al_rate"],   user_info["p_rate"] , user_info["rate_std"]]'''
    nums = len(rating_data)
    prefs = np.array([1.0]* nums).reshape(-1,1)
    for i in range(nums):
        user_id = rating_data[i][0]
        item_id = rating_data[i][1]
        rate = rating_data[i][2]
        user = user_info[user_id]
        item = item_info[item_id]
        
        # TODO: rate = rate/(1 + user[3])
        pref = (rate - user[3] - item[1]) / item[4]  
        pref =  10 * pref / np.log10(item[0] + 9) 

        # pref = (2*rate - user[1]- user[2]) * var_user/var_toal + (2*rate -  item[1]- user[2])* var_item/var_toal # TODO:需要修正和更改
        prefs[i] = round(pref)
    prefs = prefs.clip(-30,30)
    # print(max(prefs), min(prefs))  
    # print(np.unique(prefs,return_counts=True))  
    rating_data = np.hstack([rating_data,prefs])        
    return rating_data



def get_user_rateRecords_dict(rating_data_train):
    ''' dict key is user-id, item is user's rating record'''
    user_rateRecords_dict = {}
    for line in rating_data_train:
        line = list(line)
        user_id = line[0]
        if user_id in user_rateRecords_dict:
            user_rateRecords_dict[ user_id].append(line)
        else:
            user_rateRecords_dict[ user_id] = [ line ]
    return user_rateRecords_dict


def get_records(user_rateRecords_dict, user_ids):
    '''input user-ids, return those user's all rating record'''
    records = []
    for id in user_ids :
        if  id in user_rateRecords_dict:
           records += user_rateRecords_dict[id]
    return np.array(records)

def get_node_onehot_tensor(node_num):
    node_onehot = sp.eye(node_num).tocoo()
    indices =  torch.tensor(np.array([node_onehot.row, node_onehot.col]))
    v = torch.tensor(node_onehot.data).flatten()
    node_onehot_tensor = torch.sparse_coo_tensor(indices, values=v,dtype=torch.double).float()
    return node_onehot_tensor

def stat_info(rating_data, user_num, item_num):
    '''statistic info of each user and each item '''
    print("Get statistic info:",end="   ")
    # if rating_data_train not contain some users or items
    user_record_num_avg = len(rating_data) / user_num
    item_record_num_avg = len(rating_data) / item_num
    avg_rate = np.mean(rating_data[:,2])
    normal_rate = avg_rate
    rate_std = np.std(rating_data[:,2])
    p_rate = 0
    user_info_dict = {}
    item_info_dict = {}
    for line in rating_data:
        user_id, item_id, rate = line[0], line[1], line[2]
        if user_id in user_info_dict:
            user_info_dict[user_id]["record_list"].append([item_id, rate])
        else:
            user_info_dict[user_id] = {"record_list": [[item_id, rate]]}
        if item_id in item_info_dict:
            item_info_dict[item_id]["record_list"].append([user_id, rate])
        else:
            item_info_dict[item_id] = { "record_list" :[ [user_id, rate]]}
    
    for user_id in user_info_dict.keys():
        record_arr = np.array(user_info_dict[user_id]["record_list"])
        user_info_dict[user_id]["record_num"] = len( record_arr)
        user_info_dict[user_id]["avg_rate"] = np.mean( record_arr[:,1])
        user_info_dict[user_id]["rate_std"] = np.std( record_arr[:,1])
    for item_id in item_info_dict:
        record_arr = np.array(item_info_dict[item_id]["record_list"])
        item_info_dict[item_id]["record_num"] = len( record_arr)

        if len( record_arr) > 5:
            item_info_dict[item_id]["avg_rate"] = np.mean( record_arr[:,1])
        else:
            item_info_dict[item_id]["avg_rate"] = (np.mean( record_arr[:,1]) + avg_rate)/2
        # if len( record_arr) == 1:
        #     item_info_dict[item_id]["rate_std"] = rate_std
        # elif len( record_arr) < 5 or np.std( record_arr[:,1]) == 0 :
        #     item_info_dict[item_id]["rate_std"] = (rate_std +  np.std( record_arr[:,1]))/2
        # else:
        #     item_info_dict[item_id]["rate_std"] = np.std( record_arr[:,1])

        n = len( record_arr)
        temp_avg,temp_std = np.mean( record_arr[:,1]), np.std( record_arr[:,1])
        # item_info_dict[item_id]["avg_rate"] = (temp_avg * n + avg_rate)/(n + 1)
        # item_info_dict[item_id]["rate_std"] = (temp_std * (n-1) + 2*rate_std)/(n+1)
        
        k1 = 3
        k2 = 5
        item_info_dict[item_id]["avg_rate"] = (temp_avg * n + k1*avg_rate)/(n + k1)
        # item_info_dict[item_id]["rate_std"] = (temp_std * (n-1) + 4*rate_std)/(n+3)
        item_info_dict[item_id]["rate_std"] = (temp_std *n + k2*rate_std)/(n+k2)
    print("k2=", k2)



    
    for user_id in user_info_dict.keys():
        record_arr = np.array(user_info_dict[user_id]["record_list"])
        s = 0
        for item_id in record_arr[:,0]:
            s += item_info_dict[item_id]["avg_rate"]
        user_info_dict[user_id]["normal_rate"] = s / user_info_dict[user_id]["record_num"]
        user_info_dict[user_id]["p_rate"] =  user_info_dict[user_id]["avg_rate"] -  user_info_dict[user_id]["normal_rate"]
    for item_id in item_info_dict:
        record_arr = np.array(item_info_dict[item_id]["record_list"])
        s = 0
        for user_id in record_arr[:,0]:
            s += user_info_dict[user_id]["avg_rate"]
        item_info_dict[item_id]["normal_rate"] = s / item_info_dict[item_id]["record_num"]
        item_info_dict[item_id]["p_rate"] =  item_info_dict[item_id]["avg_rate"] -  item_info_dict[item_id]["normal_rate"]
    
    user_dict = {}
    item_dict = {}
    for user_id in user_info_dict:
        user_info  = user_info_dict[user_id]
        user_dict[user_id] = [ user_info["record_num"]  , user_info["avg_rate"],  user_info["normal_rate"],   user_info["p_rate"] , user_info["rate_std"], user_info["record_list"]]
        # user_dict[user_id] = [ user_info["avg_rate"],  user_info["normal_rate"],   user_info["p_rate"] ]
    for item_id in item_info_dict:
        item_info  = item_info_dict[item_id]
        item_dict[item_id] = [ item_info["record_num"] , item_info["avg_rate"],  item_info["normal_rate"],   item_info["p_rate"], item_info["rate_std"], item_info["record_list"] ]
        # item_dict[item_id] = [ item_info["avg_rate"],  item_info["normal_rate"],   item_info["p_rate"] ]


    # if rating_data_train not contain some users or items
    user_record_num_avg = len(rating_data) / user_num
    item_record_num_avg = len(rating_data) / item_num
    avg_rate = np.mean(rating_data[:,2])
    normal_rate = avg_rate
    rate_std = np.std(rating_data[:,2])
    p_rate = 0
    for user_id in range(user_num):
        if user_id not in user_info_dict:
            user_dict[user_id] = [user_record_num_avg , avg_rate, normal_rate, p_rate, rate_std,[]]
            # user_dict[user_id] = [ avg_rate, normal_rate, p_rate]

    for item_id in range(item_num):
        if item_id not in item_info_dict:
            item_dict[item_id] = [item_record_num_avg , avg_rate, normal_rate, p_rate, rate_std,[]]
            # item_dict[item_id] = [ avg_rate, normal_rate, p_rate]
    print("done")
    
    return user_dict, item_dict
        
def get_some_stat(user_info,item_info):
    item_avg ,item_std , user_bias = [], [], []
    for i in  range(len(item_info)):
        item_avg.append(item_info[i][1])
        item_std.append(item_info[i][4])
    for i in range(len(user_info)):
        user_bias.append(user_info[i][3])

    item_avg ,item_std , user_bias = np.array(item_avg) , np.array(item_std) , np.array(user_bias)

    
    return item_avg ,item_std , user_bias 

    
        
def get_info_matrix(user_info, item_info, user_id_list, item_id_list):
    user_info_m , item_info_m = [], []
    for user_id in user_id_list:
        user_info_m.append(user_info[user_id])
    for item_id in item_id_list:
        item_info_m.append(item_info[item_id])
    user_info_m , item_info_m =np.array(user_info_m ), np.array(item_info_m)
    user_info_m , item_info_m =torch.tensor(user_info_m).float() , torch.tensor(user_info_m).float()
    return  user_info_m, item_info_m
     


def add_trust_strength(trust_data,rating_data):
    n = len(trust_data)
    strength = np.array([1]*n).reshape(-1,1)
    uu_dict = {}
    for i in range(n):
        u1 = trust_data[i][0]
        u2 = trust_data[i][1]
        if u1 in uu_dict:
            uu_dict[u1].add(u2)
        else:
            uu_dict[u1] = set([u2])
        
        # if u2 in uu_dict:
        #     uu_dict[u2].add(u1)
        # else:
        #     uu_dict[u2] = set([u1])
    
    user_list = np.unique(trust_data)
    user_item_dict = {}
    for u in user_list:
        user_item_dict[u] = rating_data[rating_data[:,0]==u]


    for i in range(n):
        u1 = trust_data[i][0]
        u2 = trust_data[i][1]
        r1 = len(uu_dict[u1].intersection(uu_dict[u2])) if u1 in uu_dict and u2 in uu_dict else 0

        rating_list_1 = user_item_dict[u1]
        rating_list_2 = user_item_dict[u2]
        u1_d  = {}
        u2_d = {}
        for line in rating_list_1:
            u1_d[line[1]] = line[3]
        for line in rating_list_2:
            u2_d[line[1]] = line[3]
        
        r2 = 0
        commen_items = list(set(u1_d.keys()).intersection(set(u2_d.keys()))) 
        for item in commen_items:
            r2 += abs(u1_d[item] - u2_d[item]) < 10
        strength[i] = 2 +  r1 + r2
    strength = np.round(np.log2(strength)).clip(0,10)
    trust_data = np.hstack((trust_data,strength))
    return trust_data




def get_rmse(a, b):
    # rmse = np.sqrt( np.mean( np.square(a - b)))
    rmse = torch.sqrt( torch.mean( torch.pow(a - b,2)))

    return rmse.item()

def get_mae(a,b):
    mae = torch.mean( torch.abs(a -b))
    return mae.item()

# 学习率逐步下降
def adjust_learning_rate(opt):
    if opt != None:
        for param_group in opt.param_groups:
            '''
            调正每组的 lr
            '''
            param_group['lr'] = max(param_group['lr'] * 0.1, 0.001)
    return 


def get_recall(pred_rate, true_rate, F):
    x,y = 0,0
    for i in range(len(pred_rate)):
        if true_rate[i] >= F:
            y += 1
            if pred_rate[i] >= F:
                x += 1 
    print(len(pred_rate),x,y)
    return x/y 
    