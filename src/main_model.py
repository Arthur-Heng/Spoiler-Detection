import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import SAGEConv,GCNConv
import torch.nn.functional as F
from layer import SD_layer, SimpleHGN
import random
from utils import view_int

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      #torch.backends.cudnn.deterministic = True
# setup_seed(42)

# Const
movie0, movie_num = 7991, 1572
review0, review_num = 9570, 573913
graph1_node = 9562
graph2_node = 583482
graph3_node = 843213

# sd_layer indicate the num of SD_layer
# c_c indicate the model setting of the classification
class MVSD(torch.nn.Module):
    def __init__(self,in_dim=768, hidden_dim=128, dropout=0.3, sd_layer=2, n_layer=1, int_type=2, c_c = 1, conv_type = 1):
        super().__init__()
        self.n_sd = sd_layer
        self.c_c = c_c
        self.activation = nn.Tanh()
        self.text_relu = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.LeakyReLU()
        )
        self.meta_relu = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LeakyReLU()
        )
        self.struct_relu = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.LeakyReLU()
        )
        self.graph2_linear = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            self.activation
        )
        self.graph3_linear = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            self.activation
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,2)
        )
        self.classifier_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,2)
        )
        beta = 0.05
        self.graph2_conv = SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta)

        self.SD_layer_list = nn.ModuleList()
        self.view_int_list = nn.ModuleList()
        
        for i in range(sd_layer):
            self.SD_layer_list.append(SD_layer(conv_type = conv_type,hidden_dim = hidden_dim, dropout = dropout, n_layer = n_layer))
            self.view_int_list.append(view_int(hidden_dim = hidden_dim, int_type = int_type))
    
    def forward(self, batch):
        batch_size = batch.batch_size
        ############################ indexing the 3 subgraph
        index1 = ((batch.n_id[batch.edge_index[0]] <= graph1_node) & (batch.n_id[batch.edge_index[1]] <= graph1_node))
        edge_index1 = batch.edge_index[:,index1]
        edge_type1 = batch.edge_type[index1]
        index2 = ((batch.n_id[batch.edge_index[0]] >= movie0) & (batch.n_id[batch.edge_index[1]] >= movie0)
                                        & (batch.n_id[batch.edge_index[0]] <= graph2_node) & (batch.n_id[batch.edge_index[1]] <= graph2_node))
                                    
        edge_index2 = batch.edge_index[:,index2]
        edge_type2 = batch.edge_type[index2]
        index3 = ((batch.n_id[batch.edge_index[0]] >= review0) & (batch.n_id[batch.edge_index[1]] >= review0))
        edge_index3 = batch.edge_index[:,index3]
        edge_type3 = batch.edge_type[index3]
        meta = batch.x[:, 768:768+5]
        text = batch.x[:,:768]
        struct = batch.x[:,-768:]
        ########################## 768 --> 256,  5 --> 64
        text = self.text_relu(text)
        meta = self.meta_relu(meta)
        struct = self.struct_relu(struct)
        #########################
        sub1_text = text
        sub1_struct = struct

        sub2_text = text
        sub2_meta = meta

        sub3_text = text
        sub3_meta = meta

        movie_id = (batch.n_id >= movie0) & (batch.n_id < movie0 + movie_num) # attain the indexes of movie
        num = movie_id.sum()
        for i in range(self.n_sd):
            sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta = self.SD_layer_list[i](sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta, edge_index1, edge_index2, edge_index3, edge_type1, edge_type2, edge_type3)
            sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta = self.view_int_list[i](sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta, movie_id, batch_size, num)
        
        if self.c_c == 1:
            sub2_feat_rev = self.graph2_linear(torch.cat((sub2_text[:batch_size], sub2_meta[:batch_size]),1))
            sub3_feat_rev = self.graph3_linear(torch.cat((sub3_text[:batch_size], sub3_meta[:batch_size]),1))
            x = self.classifier(torch.cat((sub2_feat_rev, sub3_feat_rev),1))
            
        elif self.c_c == 2:
            sub2_feat_rev = self.activation(self.graph2_conv(sub2_text, edge_index2, edge_type2))
            x = self.classifier_conv(sub2_feat_rev)
        
        return x
