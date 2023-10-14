import numpy as np
import dgl, torch
import scipy.sparse as sp


def rating_graph_gen(rating_data_train, user_num, item_num):
    # construct rating graph
    rate_data_mat = sp.coo_matrix((rating_data_train[:,3], (rating_data_train[:,0], rating_data_train[:,1] + user_num)),
                                    shape = (user_num + item_num, user_num + item_num))

    rate_data_mat = rate_data_mat.T + sp.eye(user_num + item_num)*30 # self-loop and bi-directed graph
    # rate_data_mat = rate_data_mat + rate_data_mat.T + sp.eye(user_num + item_num)*7 # self-loop and bi-directed graph

    x,y = rate_data_mat.nonzero()
    v = rate_data_mat.data
    rate_g  = dgl.graph((x,y),num_nodes= user_num + item_num)
    # rate_g.edata['r'] = torch.tensor(v,dtype=torch.int32)    
    rate_g.edata['p'] = torch.tensor(v,dtype=torch.int32)    

    return rate_g

def rating_graph_gen_new(rating_data_train, user_num, item_num):
    # construct rating graph
    v = rating_data_train[:,2]
    # np.random.shuffle(v[100000:])
    rate_data_mat = sp.coo_matrix((v, (rating_data_train[:,0], rating_data_train[:,1] )),
                                    shape = (user_num ,item_num))
    user_eye = sp.eye(user_num) * 7
    item_eys = sp.eye(item_num) * 7
    mat = sp.vstack([ sp.hstack([rate_data_mat, user_eye]), sp.hstack([item_eys, rate_data_mat.T])])
    # mat = sp.vstack([  sp.hstack([item_eys, rate_data_mat.T]),sp.hstack([rate_data_mat, user_eye])])
    # mat = sp.vstack([ sp.hstack([user_eye,rate_data_mat]), sp.hstack([rate_data_mat.T,item_eys])])

    # mat = mat + sp.eye(user_num + item_num)* 7

    # rate_data_mat = rate_data_mat + rate_data_mat.T + sp.eye(user_num + item_num)*7 # self-loop and bi-directed graph

    x,y = mat.nonzero()
    v = mat.data
    rate_g  = dgl.graph((x,y),num_nodes= user_num + item_num)
    rate_g.edata['r'] = torch.tensor(v,dtype=torch.int32)    

    return rate_g
    


def trust_graph_gen(trust_data, user_num):
    # trust_v = [1] * len(trust_data)
    trust_data_mat = sp.coo_matrix((trust_data[:,2], (trust_data[:,0], trust_data[:,1])),
                                    shape = (user_num, user_num))
    
    trust_data_mat = trust_data_mat.T + sp.eye(user_num) * 10 # self-loop and bi-directed graph
    
    x, y  = trust_data_mat.nonzero()
    v = trust_data_mat.data.clip(0,10)

    trust_g = dgl.graph((x,y), num_nodes= user_num)
    trust_g.edata['r'] = torch.tensor(v,dtype=torch.int32)
    return trust_g

def union_graph_gen(rating_data_train, trust_data, user_num, item_num):
    print("Generate graph:",end=" ")
    trust_v = [3.5] * len(trust_data)
    trust_data_mat = sp.coo_matrix((trust_v, (trust_data[:,0], trust_data[:,1])),
                                    shape = (user_num, user_num))

    trust_data_mat = trust_data_mat + trust_data_mat.T # rate_g had self-loop, so trust_g remove it 

    rate_data_mat = sp.coo_matrix((rating_data_train[:,2], (rating_data_train[:,0], rating_data_train[:,1] + user_num)),
                                    shape = (user_num + item_num, user_num + item_num))

    rate_data_mat = rate_data_mat + rate_data_mat.T + sp.eye(user_num + item_num)* 7 # self-loop and bi-directed graph

    x1, y1  = trust_data_mat.nonzero()
    v1 = trust_data_mat.data

    x2,y2 = rate_data_mat.nonzero()
    v2 = rate_data_mat.data

    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    v = np.hstack((v1,v2))

    union_g =  dgl.graph((x, y),num_nodes= user_num + item_num)
    union_g.edata['r'] = torch.tensor(v)
    print("done")
    
    return union_g


def graph_add_aug_r(g, user_info,item_info):
    print("Add aug_r to graph:",end="    ")
    edges_src, edges_dst = g.edges()
    edges_num = len(edges_src)
    user_num = len(user_info)
    aug_r = []
    for i in range(edges_num):
        rate = g.edata["r"][i].item()
        node_id1 = edges_src[i].item()
        node_id2 = edges_dst[i].item()
        node1_info, node2_info = [], []
        if node_id1 < user_num:
            node1_info = user_info[node_id1]
        else:
            node1_info = item_info[node_id1 - user_num]
        
        if node_id2 < user_num:
            node2_info = user_info[node_id2]
        else:
            node2_info = item_info[node_id2 - user_num]
        
        edge_info = [rate] + node1_info + node2_info
        aug_r.append(edge_info)
    aug_r = torch.tensor( np.array(aug_r) ).reshape(edges_num,-1)
    g.edata["aug_r"] = aug_r
    print("done")
    return g