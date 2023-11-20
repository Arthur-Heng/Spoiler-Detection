import torch
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.utils import softmax
import torch.nn.functional as F
import torch.nn as nn

class SimpleHGN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim=200, beta=None, final_layer=False):
        super(SimpleHGN, self).__init__(aggr = "add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                #torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)
            
        # return output, self.alpha.detach() # What is alpha
        return output
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)
        return out

    def update(self, aggr_out):
        return aggr_out

class GatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GatedRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.RGCN1 = RGCNConv(
            in_channels=out_channels, out_channels=out_channels, num_relations=num_relations)
        self.attention_layer = nn.Linear(2 * out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(
            self.attention_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, node_features, edge_index, edge_type):

        u_0 = self.RGCN1(node_features, edge_index, edge_type)
        a_1 = self.sigmoid(self.attention_layer(
            torch.cat((u_0, node_features), dim=1)))
        h_1 = self.tanh(u_0) * a_1 + node_features * (1 - a_1)

        return h_1

# set the meta_hidden to be the same as the others
class SD_layer(torch.nn.Module):
    def __init__(self,conv_type=1, hidden_dim=256, dropout=0.3, n_layer = 1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.LeakyReLU()

        # graph conv for text
        self.text1_conv = nn.ModuleList()   # sub graph 1
        self.text2_conv = nn.ModuleList()   # sub graph 2
        self.text3_conv = nn.ModuleList()   # sub graph 3

        # graph conv for struct
        self.struct1_conv = nn.ModuleList()  # sub graph 1

        # graph conv for meta   
        self.meta2_conv = nn.ModuleList()   # sub graph 2
        self.meta3_conv = nn.ModuleList()   # sub graph 3

        for i in range(n_layer):
            if conv_type == 1:
                beta = 0.05
                self.text1_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))
                self.text2_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))
                self.text3_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))

                self.struct1_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))
                self.meta2_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))
                self.meta3_conv.append(SimpleHGN(num_edge_type=10, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta))

            if conv_type == 2:
                self.text1_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.text2_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.text3_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))

                self.struct1_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.meta2_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.meta3_conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))

            if conv_type == 3:
                self.text1_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.text2_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.text3_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))

                self.struct1_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.meta2_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))
                self.meta3_conv.append(GatedRGCN(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=10))

    def forward(self, sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta, edge_index1, edge_index2, edge_index3, edge_type1, edge_type2, edge_type3):
        
        # sub graph 1
        for layer in self.text1_conv:
            sub1_text = self.dropout(self.activation(layer(sub1_text, edge_index1, edge_type1)))
        for layer in self.struct1_conv:
            sub1_struct = self.dropout(self.activation(layer(sub1_struct, edge_index1, edge_type1)))

        # sub graph 2
        for layer in self.text2_conv:
            sub2_text = self.dropout(self.activation(layer(sub2_text, edge_index2, edge_type2)))
        for layer in self.meta2_conv:
            sub2_meta = self.dropout(self.activation(layer(sub2_meta, edge_index2, edge_type2)))

        # sub graph 3
        for layer in self.text3_conv:
            sub3_text = self.dropout(self.activation(layer(sub3_text, edge_index3, edge_type3)))
        for layer in self.meta3_conv:
            sub3_meta = self.dropout(self.activation(layer(sub3_meta, edge_index3, edge_type3)))
        
        return sub1_text, sub1_struct, sub2_text, sub2_meta, sub3_text, sub3_meta
