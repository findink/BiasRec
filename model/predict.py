from os import pread
import torch.nn as nn
import torch
    

class prefPredictLayer(nn.Module):
    '''
     id --> onehot --MLP--> embed --EGAT--> embed --GAT--> embed
    '''
    def __init__(self,  hide_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hide_dims*4, hide_dims),  # 输入用户和item的向量。
            # nn.BatchNorm1d(hide_dims),
            nn.ReLU(),
            nn.Linear(hide_dims, 5), 
            # nn.BatchNorm1d(5),
            # nn.Sigmoid()，
            nn.ReLU(),
            nn.Linear(5,1),
            # nn.Tanh() 
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hide_dims*3, hide_dims*1),  # 输入用户和item的向量。
            nn.ReLU(),
            nn.Linear(hide_dims, 1), 
            nn.ReLU() 
        )
        
        info_dims = 5
        self.info_mlp = nn.Sequential(
            nn.Linear(hide_dims*5 + info_dims *2, hide_dims*1),  # 输入用户和item的向量。
            nn.ReLU(),
            nn.Linear(hide_dims, 1), 
            nn.ReLU()
        )
        self.info_mlp2 = nn.Sequential(
            nn.Linear( info_dims *2, info_dims * 2 ),  # 输入用户和item的向量。
            nn.ReLU(),
            nn.Linear(info_dims * 2, 6), 
            # nn.Sigmoid()
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear( 14, 8 ),  # 输入用户和item的向量。
            nn.ReLU(),
            nn.Linear(8, 1), 
            nn.ReLU()
        )
        
        

    
    def forward(self, node_embeds1, node_embeds2):
        # node_embeds1 = self.trans(node_embeds1)
        hide_dims = 32
        # tensor = torch.cat((node_embeds1[:,:hide_dims*2], node_embeds2), dim=1)
        tensor = torch.cat((node_embeds1, node_embeds2), dim=1)
        pred = self.mlp(tensor) 
        # pred = self.final(embed)
        # pred = self.info_mlp(tensor)
        # pred = self.info_mlp2(tensor)  * 5
        # pred = self.mlp2(tensor)
        # pred = pred.clamp(1,5)     

        return pred

# class PrefPredictLayer(nn.Module):
#     def __init__(self,  hide_dims):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hide_dims*2, hide_dims*1),  # 输入用户和item的向量。
#             nn.ReLU(),
#             nn.Linear(hide_dims, 1), 
#             # nn.Sigmoid()，
#             nn.ReLU()
#         )
    
#     def forward(self, node_embeds1, node_embeds2):
#         # node_embeds1 = self.trans(node_embeds1)
#         tensor = torch.cat((node_embeds1, node_embeds2), dim=1)
#         prefs = self.mlp(tensor)
#         return prefs