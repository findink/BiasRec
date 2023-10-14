import scipy.io as scio
import numpy as np
import dgl, torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class NodeEmbed(nn.Module):
    '''
     id --> onehot --MLP--> embed --EGAT--> embed --GAT--> embed
    '''
    def __init__(self, nodeNum, hide_dims, rate_g = None , trust_g = None):
        super(NodeEmbed, self).__init__()
         
        # 对 one-hot 进行变换 （包括 user, item)
        self.w = nn.Linear(nodeNum, hide_dims, bias = False)
        # self.w = nn.Linear(nodeNum, hide_dims)
        

        self.num_heads = 3
        self.hide_dims = hide_dims
        self.rate_g = rate_g
        self.trust_g = trust_g
        self.pref_embeding = nn.Embedding(61,10)  # 太多了，10就最够了
        self.trust_embeding = nn.Embedding(11,10)
        self.egat = dgl.nn.EGATConv( in_node_feats  = hide_dims,  # 。。。
                                     in_edge_feats  = 10,
                                     out_node_feats = hide_dims,
                                     out_edge_feats = 10,
                                     num_heads = self.num_heads)  
        self.egat2 = dgl.nn.EGATConv( in_node_feats  = hide_dims * 2,
                                     in_edge_feats  = 10,
                                     out_node_feats = hide_dims * 2,
                                     out_edge_feats = 10,
                                     num_heads = self.num_heads)  
        self.gat = dgl.nn.GATConv(hide_dims * 2, hide_dims *2,num_heads = self.num_heads)
        self.embeds_user_agg = nn.Linear(hide_dims * 3, hide_dims)
        self.embeds_item_agg = nn.Linear(hide_dims * 2, hide_dims)
        # self.act1 = torch.nn.LeakyReLU()
        self.act1 = nn.ReLU()
        self.act2 = nn.LeakyReLU()
        
        
        initializer = nn.init.xavier_uniform_
        # initializer = nn.init.kaiming_normal_
        initializer(self.w.weight) 
        initializer(self.pref_embeding.weight)
        initializer(self.trust_embeding.weight)
    
    def forward(self, node_onehot_tensor, use_aug_r = False):
        embeds_base = self.w(node_onehot_tensor)
       
        # use_aug_r = True
        if use_aug_r:
            edge_rates = self.rate_g.edata["aug_r"].float()
            embeds_egat, _ =  self.egat2(self.rate_g, embeds_base, edge_rates)  # shape: node_num, head_num,embed
            embeds_egat = torch.div(torch.sum(embeds_egat, dim=1), self.num_heads)
            _ = torch.div(torch.sum(_, dim=1), self.num_heads)
            embeds_egat = self.act1(embeds_egat)
        else:
            edge_prefs = self.rate_g.edata["p"].reshape(-1,1).int() + 30
            # print(max(edge_prefs),min(edge_prefs))
            # return
            edge_embeding = self.pref_embeding(edge_prefs).view(-1,10)
            embeds_egat, _ =  self.egat(self.rate_g, embeds_base, edge_embeding)  # shape: node_num, head_num,embed
            embeds_egat = torch.div(torch.sum(embeds_egat, dim=1), self.num_heads)
            embeds_egat = self.act1(embeds_egat)
            # self.rate_g.ndata["ft"] = embeds_base
            # self.rate_g.edata['a'] = dgl.ops.edge_softmax(self.rate_g, self.rate_g.edata['r'].float())
            # self.rate_g.update_all(dgl.function.u_mul_e("ft","a","m"), dgl.function.sum("m", "ft"))
            # embeds_egat =  self.rate_g.ndata['ft']

        user_num = self.trust_g.num_nodes()
        embeds_base_user = embeds_base[:user_num]
        embeds_base_item = embeds_base[user_num:]

        embeds_egat_user = embeds_egat[:user_num]
        embeds_egat_item = embeds_egat[user_num:]

        # embeds_trust_user = self.gat(self.trust_g, embeds_base_user)
        # embeds_trust_user = torch.div(torch.sum(embeds_trust_user, dim=1), self.num_heads)
        # embeds_trust_user = self.act2(embeds_trust_user)
        # user_embeds = torch.cat([embeds_base_user,  embeds_egat_user, embeds_trust_user], dim=1)
        # embeds_base_user = F.normalize(embeds_base_user,dim=1)
        # embeds_egat_user = F.normalize(embeds_egat_user,dim=1)
        
        embeds_egat_user = embeds_egat_user + embeds_base_user
        embeds_egat_item = embeds_egat_item + embeds_base_item
        user_embeds = torch.cat([embeds_base_user,  embeds_egat_user], dim=1)
        item_embeds = torch.cat([embeds_base_item,  embeds_egat_item], dim=1)

        # embeds_trust_user = self.gat(self.trust_g, user_embeds)
        # embeds_trust_user = torch.div(torch.sum(embeds_trust_user, dim=1), self.num_heads)
        # embeds_trust_user = self.act2(embeds_trust_user)
        # print("here0")
        # edge_trust = self.trust_g.edata["r"].reshape(-1,1).int()
        # print(min(edge_trust),max(edge_trust))
        edge_trust = self.trust_g.edata["r"].reshape(-1,1).int()
        trust_edge_embeding = self.trust_embeding(edge_trust).view(-1,10)
        embeds_trust_user, _ =  self.egat2(self.trust_g, user_embeds, trust_edge_embeding)  # shape: node_num, head_num,embed
        embeds_trust_user = torch.div(torch.sum(embeds_trust_user, dim=1), self.num_heads)
        embeds_trust_user = self.act2(embeds_trust_user)
    
        # user_embeds = F.normalize(user_embeds,dim=1)
        # embeds_trust_user = F.normalize(embeds_trust_user,dim=1)
        # item_embeds = F.normalize(item_embeds,dim=1)


        return user_embeds, embeds_trust_user, item_embeds
        # return embeds_base_user,embeds_trust_user, embeds_base_item
        # return embeds_egat_user, embeds_egat_item



