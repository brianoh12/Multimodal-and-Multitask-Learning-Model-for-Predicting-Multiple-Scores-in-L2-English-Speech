import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
import math

##############################################################
class CustomAttention_pooling(nn.Module):
    def __init__(self, input_dim):
        super(CustomAttention_pooling, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)

        return context, attn
    

class CustomAttention(nn.Module):
    def __init__(self, input_dim):
        super(CustomAttention, self).__init__()
        # Attention 구현에 필요한 레이어 정의
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

    def forward(self, target, non_target):
        # 주목 메커니즘 구현
        query = self.query_layer(target)
        key = self.key_layer(non_target)
        value = self.value_layer(non_target)

        # query와 key 사이의 유사도 계산
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 가중치 적용된 값 계산
        attention_output = torch.matmul(attention_weights, value)
        return attention_output
    

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a,= hyp_params.orig_d_l, hyp_params.orig_d_a,
        self.d_l, self.d_a, = 200, 200, 
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.output_dim = 5
        self.lstm_units = 100
        # 추가적인 특성을 위한 플레이스홀더
        self.acoustic_feature_dim = 88
        self.lexical_feature_dim = 33
        self.syntactic_feature_dim = 23
      
        
        
        concat_dim = self.lstm_units + self.acoustic_feature_dim + self.lexical_feature_dim + self.syntactic_feature_dim
        

        self.partial_mode = self.lonly + self.aonly 
        if self.partial_mode == 1:
            combined_dim = self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = (self.d_l + self.d_a)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_a_with_l = self.get_network(self_type='al')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        # self.trans_l_mem = nn.LSTM(self.d_l, self.d_l, 1)
        # self.trans_a_mem = nn.LSTM(self.d_a, self.d_a, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
   
        self.lstm = nn.LSTM(input_size=self.d_l+self.d_a, hidden_size=self.lstm_units, batch_first=True)
        
        self.transform_layers = nn.ModuleList([nn.Linear(self.lstm_units, self.lstm_units) for _ in range(self.output_dim)])
        
        self.attention_pooling = CustomAttention_pooling(self.lstm_units)
        
        self.attention = CustomAttention(concat_dim)
        self.fc = nn.Linear(concat_dim * 2, 1)   


        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_a, acoustic_feature, lexical_feature,syntactic_feature):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)


        # (A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
        # h_ls = torch.cat([h_l_with_as, h_l_with_as], dim=2)
        h_ls = self.trans_l_mem(h_l_with_as)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        # (L) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        # h_as = torch.cat([h_a_with_ls, h_a_with_ls], dim=2)
        h_as = self.trans_a_mem(h_a_with_ls)
        # h_as = h_a_with_ls
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]
        
        last_hs = torch.cat([last_h_l, last_h_a], dim=1)
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        last_hidden = self.lstm(last_hs_proj)
        lstm_outs = []
        for i in range(self.output_dim):
            transformed = self.transform_layers[i](last_hidden[0])
            lstm_outs.append(transformed)

        final_feature = [self.attention_pooling(i)[0] for i in lstm_outs]
        es_pr_feat_concat = [torch.cat([rep, acoustic_feature, lexical_feature, syntactic_feature], dim=-1)
                             for rep in final_feature]
        
        pos_avg_hz_lstm = torch.cat([rep.unsqueeze(1) for rep in es_pr_feat_concat], dim=1)
        final_preds = []
        attention_concats = []
        for index in range(self.output_dim):
            mask = torch.ones(self.output_dim, dtype=torch.bool)
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:,mask,:]
            target_rep = pos_avg_hz_lstm[:, index:index+1, :]

            att_attention = self.attention(target_rep, non_target_rep)
            attention_concat = torch.cat([target_rep, att_attention], dim=-1)
            attention_concat = attention_concat.view(attention_concat.size(0), -1)  # Flatten
            final_pred = self.fc(attention_concat)
            attention_concats.append(attention_concat)
            final_preds.append(final_pred)

        return torch.cat(final_preds, dim=-1), torch.cat(attention_concats, dim=-1)
