import torch
import torch.nn as nn
import random
import numpy as np

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))

# Interaction
class cross_attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.activation = nn.Tanh()
        self.linear_1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )
        self.linear_2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_1, input_2):
        w1 = torch.mean(self.linear_1(input_1))
        w2 = torch.mean(self.linear_2(input_2))
        w1, w2 = self.softmax(torch.Tensor([w1,w2]))
        output = w1*input_1 + w2*input_2

        return output

# Interaction
class cross_attention_1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.activation = nn.Tanh()
        self.linear_1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )
        self.linear_2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_1, input_2):
        w1 = torch.mean(self.linear_1(input_1))
        w2 = torch.mean(self.linear_2(input_2))
        w1, w2 = self.softmax(torch.Tensor([w1,w2]))
        output = w2*input_1 + w1*input_2

        return output
    
class view_int(nn.Module):
    def __init__(self, hidden_dim, int_type=1):
        super().__init__()
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.int_type = int_type

        if self.int_type == 1:
            self.graph1_linear = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                self.activation
            )
            self.graph2_linear = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                self.activation
            )
            self.graph3_linear = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                self.activation
            )
            self.int2_attn1_mov = cross_attention(hidden_dim=hidden_dim)
            self.int2_attn2_rev = cross_attention(hidden_dim=hidden_dim)

        if self.int_type == 2:
            self.int1_attn1_mov = cross_attention(hidden_dim=hidden_dim)
            self.int1_attn2_rev = cross_attention(hidden_dim=hidden_dim)
            self.int2_attn1_mov = cross_attention(hidden_dim=hidden_dim)
            self.int2_attn2_rev = cross_attention(hidden_dim=hidden_dim)
            
        if self.int_type == 3:
            self.int1_attn1_mov = cross_attention_1(hidden_dim=hidden_dim)
            self.int1_attn2_rev = cross_attention_1(hidden_dim=hidden_dim)
            self.int2_attn1_mov = cross_attention_1(hidden_dim=hidden_dim)
            self.int2_attn2_rev = cross_attention_1(hidden_dim=hidden_dim)

    def forward(self, sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta, movie_id, batch_size, num):
        # different view interaciton
        if self.int_type == 1:
            self.num = num
            self.movie_id = movie_id
            self.batch_size = batch_size
            
            sub1_feat_mov = self.graph1_linear(torch.cat((sub1_text[self.movie_id], sub1_struct[self.movie_id]),1))
            sub2_feat_mov = self.graph2_linear(torch.cat((sub2_text[self.movie_id], sub2_meta[self.movie_id]),1))
            sub2_feat_rev = self.graph2_linear(torch.cat((sub2_text[:self.batch_size], sub2_meta[:self.batch_size]),1))
            sub3_feat_rev = self.graph3_linear(torch.cat((sub3_text[:self.batch_size], sub3_meta[:self.batch_size]),1))

        if self.int_type == 2:
            self.num = num
            self.movie_id = movie_id
            self.batch_size = batch_size
            sub1_feat_mov = self.int1_attn1_mov(sub1_text[self.movie_id], sub1_struct[self.movie_id])
            sub2_feat_mov = self.int1_attn1_mov(sub2_text[self.movie_id], sub2_meta[self.movie_id])
            sub2_feat_rev = self.int1_attn2_rev(sub2_text[:self.batch_size], sub2_meta[:self.batch_size])
            sub3_feat_rev = self.int1_attn2_rev(sub3_text[:self.batch_size], sub3_meta[:self.batch_size])

        if self.int_type == 3:
            self.num = num
            self.movie_id = movie_id
            self.batch_size = batch_size
            sub1_feat_mov = self.int1_attn1_mov(sub1_text[self.movie_id], sub1_struct[self.movie_id])
            sub2_feat_mov = self.int1_attn1_mov(sub2_text[self.movie_id], sub2_meta[self.movie_id])
            sub2_feat_rev = self.int1_attn2_rev(sub2_text[:self.batch_size], sub2_meta[:self.batch_size])
            sub3_feat_rev = self.int1_attn2_rev(sub3_text[:self.batch_size], sub3_meta[:self.batch_size])

        # sub graph interaction
        movie_mix = self.int2_attn1_mov(sub1_feat_mov, sub2_feat_mov)
        review_mix = self.int2_attn2_rev(sub2_feat_rev, sub3_feat_rev)

        sub1_text_new, sub1_struct_new, sub2_text_new , sub2_meta_new, sub3_text_new, sub3_meta_new = sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta
        sub1_text_new[self.movie_id] = movie_mix
        sub1_struct_new[self.movie_id] = movie_mix
        sub2_text_new[self.movie_id] = movie_mix
        sub2_text_new[:self.batch_size] = review_mix
        sub2_meta_new[self.movie_id] = movie_mix
        sub2_meta_new[:self.batch_size] = review_mix
        sub3_text_new[:self.batch_size] = review_mix
        sub3_meta_new[:self.batch_size] = review_mix

        # return sub1_text, sub1_struct, sub2_text , sub2_meta, sub3_text, sub3_meta
        return sub1_text_new, sub1_struct_new, sub2_text_new , sub2_meta_new, sub3_text_new, sub3_meta_new