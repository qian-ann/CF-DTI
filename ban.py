import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import math
import numpy as np


class FocalAtt(nn.Module):
    def __init__(self, __C,q_dim,k_dim,mode="CA"):
        super(FocalAtt, self).__init__()
        assert __C.hid_dim % __C.n_heads == 0
        if mode=="CA":
            self.CA=SorCA(__C,q_dim,k_dim)
        else:
            self.CA=SCA(__C,q_dim,k_dim)

    def focal(self,q,attention,per):
        bsz,_,q_dim = q.shape
        # 根据focal情况，最多保留per的比例
        lenmax = int(per * q.shape[1])
        # batch_size x query_len x key_len  -->   batch_size x query_len
        att = attention.mean(dim=-1)
        # att, _ = attention.max(dim=-1)
        ## focal part ## batch_size x query_len
        funcF = att * torch.sum(att > 0, dim=-1, keepdim=True) - torch.sum(att, dim=-1, keepdim=True)
        focal_att = torch.where(funcF > 0, torch.ones_like(att), torch.zeros_like(att))
        att = nn.functional.softmax(att * focal_att, -1)
        p_len = torch.sum(att > 0, dim=-1)
        p_len[p_len > lenmax] = lenmax
        p_len_Max = p_len.max()
        attidx = (torch.argsort(att, dim=1, descending=True)).int()[:, :p_len_Max]
        for idx in range(bsz):
            vv0 = q[idx, attidx[idx, :p_len[idx]], :].unsqueeze(0)
            vv0 = torch.cat([vv0, torch.zeros([1, p_len_Max - p_len[idx], q_dim],requires_grad=True).to(q.device)],1)
            # vv0 = att[idx, attidx[idx, :]].unsqueeze(0).unsqueeze(-1) * vv0
            if idx == 0:
                vv = vv0
            else:
                vv = torch.cat([vv, vv0], dim=0)

        return vv, attidx

    def forward(self, q, k, per=0.2, PE=True, SF=False, outmode=None):
        # attention = [batch_size, seq_len_q, seq_len_K]  未经过softmax
        q, attention, feature =self.CA(q, k, k, PE, SF, outmode)

        attidx_q=None
        attidx_k=None
        if outmode!=None and 'focal' in outmode:
            # q, attidx_q=self.focal(q,attention,per)
            k, attidx_k=self.focal(k,attention.transpose(1,2),per)


        return feature, q, k, attention, attidx_q, attidx_k

class SCAtt(nn.Module):
    def __init__(self, __C, n_layer,q_dim,k_dim):
        super(SCAtt, self).__init__()
        self.n_layer=n_layer
        if self.n_layer > 0:
            assert __C.hid_dim % __C.n_heads == 0
            self.dec_list = nn.ModuleList([])
            if n_layer-1>0:
                self.dec_list = nn.ModuleList([SCA(__C,q_dim,k_dim)])
                self.dec_list = self.dec_list.append(nn.ModuleList([SCA(__C,__C.hid_dim,__C.hid_dim) for _ in range(n_layer-2)]))
                self.dec_end = SCA(__C, __C.hid_dim, __C.hid_dim)
            else:
                self.dec_end=SCA(__C,q_dim,k_dim)

    def forward(self, q, k, PE=True, SF=True, outmode=None):
        # 位置编码
        att=None
        out=None
        if self.n_layer>0:
            for dec in self.dec_list:
                q, att, out = dec(q, k, k, PE, SF, "atted")
            query, att, out = self.dec_end(q, k, k, PE, SF, outmode)
        return q, att, out

class SorCAtt(nn.Module):
    def __init__(self, __C, n_layer,q_dim,k_dim):
        super(SorCAtt, self).__init__()
        self.n_layer=n_layer
        if self.n_layer > 0:
            assert __C.hid_dim % __C.n_heads == 0
            self.dec_list = nn.ModuleList([])
            if n_layer-1>0:
                self.dec_list = nn.ModuleList([SorCA(__C,q_dim,k_dim)])
                self.dec_list = self.dec_list.append(nn.ModuleList([SorCA(__C,__C.hid_dim,__C.hid_dim) for _ in range(n_layer-2)]))
                self.dec_end = SorCA(__C, __C.hid_dim, __C.hid_dim)
            else:
                self.dec_end=SorCA(__C,q_dim,k_dim)

    def forward(self, q, k, PE=True, SF=True, outmode=None):
        # 位置编码
        att=None
        out=None
        if self.n_layer>0:
            for dec in self.dec_list:
                q, att, out = dec(q, k, k, PE, SF, "atted")
            q, att, out = self.dec_end(q, k, k, PE, SF, outmode)

        return q, att, out

# -------------------------------
# ---- Self or Cross Attention ----
# -------------------------------

class SorCA(nn.Module):
    def __init__(self, __C, q_dim, k_dim):
        super(SorCA, self).__init__()

        self.mhatt = MHAtt(q_dim,k_dim,__C.hid_dim,__C.n_heads,__C.dropout)
        self.ffn = FFN(__C.hid_dim,__C.mid_size,__C.out_size,__C.dropout)
        self.dropout = nn.Dropout(__C.dropout)
        self.norm = LayerNorm(__C.hid_dim)

        self.PosEnc_q = PositionalEncoding(q_dim, __C.dropout)
        self.PosEnc_k = PositionalEncoding(k_dim, __C.dropout)

    def forward(self, q, k, v, PE='', SF=True, outmode=None):
        # 位置编码
        q0=q.clone()
        if 'q' in PE:
            q=self.PosEnc_q(q)
        if 'k' in PE:
            k=self.PosEnc_k(k)
        #CAtt
        qq, att, out=self.mhatt(q, k, v, q0, SF, outmode)
        if outmode!=None and "atted" in outmode:
            #FFN
            qq = self.norm(qq + self.dropout(self.ffn(qq)))

        return qq, att, out

# -------------------------------
# ---- Self Cross Attention ----
# -------------------------------

class SCA(nn.Module):
    def __init__(self, __C,q_dim,k_dim):
        super(SCA, self).__init__()

        self.mhatt1 = MHAtt(q_dim, q_dim, __C.hid_dim,__C.n_heads,__C.dropout)
        self.mhatt2 = MHAtt(__C.hid_dim, k_dim, __C.hid_dim,__C.n_heads,__C.dropout)
        self.ffn = FFN(__C.hid_dim,__C.mid_size,__C.out_size,__C.dropout)

        self.dropout = nn.Dropout(__C.dropout)
        self.norm = LayerNorm(__C.hid_dim)

        self.PosEnc_q1 = PositionalEncoding(q_dim, __C.dropout)
        self.PosEnc_q2 = PositionalEncoding(__C.hid_dim, __C.dropout)
        self.PosEnc_k = PositionalEncoding(k_dim, __C.dropout)

    def forward(self, q, k, v, PE='', SF=True, outmode=None):
        # 位置编码
        q0=q.clone()
        if 'q' in PE:
            q=self.PosEnc_q1(q)
        #SAtt
        q0 = self.mhatt1(q, q, q0, q0, SF,"atted")[0]
        if 'q' in PE:
            q=self.PosEnc_q2(q0)
        if 'k' in PE:
            k=self.PosEnc_k(k)
        #CAtt
        qq, att, out=self.mhatt2(q, k, v, q0, SF, outmode)
        if "atted" in outmode:
            #FFN
            qq = self.norm(qq + self.dropout(self.ffn(qq)))

        return qq, att, out



# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, q_dim, k_dim, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()
        self.n_heads = n_heads
        self.hid_dim_head = hid_dim//n_heads
        assert hid_dim % n_heads == 0

        self.linear_v = nn.Linear(k_dim,hid_dim)
        self.linear_v0 = nn.Linear(k_dim,hid_dim)
        self.linear_q0 = nn.Linear(q_dim,hid_dim)
        self.Att_Map=AttM(q_dim, k_dim, hid_dim, n_heads,dropout)

        self.norm_q = LayerNorm(hid_dim)
        self.linear_merge = nn.Linear(hid_dim,hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hid_dim)

    # q，k用于生成attention矩阵，v用于生成新的值，q0和v用于bilinear
    def forward(self, q, k, v, q0, SF=True, outmode=None):
        n_batches = q.size(0)
        #生成attention矩阵
        scores=self.Att_Map(q, k)
        if SF:
            att_map = F.softmax(scores, dim=-1)
        else:
            att_map = scores
        att_map = self.dropout(att_map)
        q0 = self.linear_q0(q0)

        if outmode!=None and "atted" in outmode:
            v1 = self.linear_v(v).view(n_batches,-1,self.n_heads,self.hid_dim_head).transpose(1, 2)
            atted = torch.matmul(att_map, v1)
            atted = atted.transpose(1, 2).contiguous().view(n_batches,-1,self.hid_dim_head*self.n_heads)
            # atted = self.norm_q(q+self.dropout(self.linear_merge(atted)))
            atted = self.norm_q(q0+self.dropout(atted))
        else:
            atted = q

        x=None
        if outmode!=None and 'bilinear' in outmode:
            q0 = q0.view(n_batches, -1, self.n_heads, self.hid_dim_head).transpose(1, 2)
            v0 = self.linear_v0(v).view(n_batches, -1, self.n_heads, self.hid_dim_head).transpose(1, 2)
            # x = [batch_size, n_heads, hid_dim // n_heads]
            x = torch.einsum('bhvk,bhvq,bhqk->bhk', (q0, att_map, v0))
            # x = [batch_size, hid_dim]
            x = x.view(n_batches, -1)
            x = self.norm(x)

        # att_map: [batch_size, n_heads, seq_len_q, seq_len_K]-->[batch_size, seq_len_q, seq_len_K]
        scores=F.relu(scores,inplace=True).sum(dim=1)/ scores.shape[1]

        return atted, scores, x

class AttM(nn.Module):
    def __init__(self, q_dim, k_dim, hid_dim, n_heads,dropout):
        super(AttM, self).__init__()
        self.n_heads = n_heads
        self.hid_dim_head = hid_dim//n_heads
        assert hid_dim % n_heads == 0
        self.linear_q = nn.Linear(q_dim,hid_dim)
        self.linear_k = nn.Linear(k_dim,hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k):
        n_batches = q.size(0)
        k = self.linear_k(k)
        q = self.linear_q(q)
        q = q.view(n_batches, -1, self.n_heads, self.hid_dim_head).transpose(1, 2)
        k = k.view(n_batches, -1, self.n_heads, self.hid_dim_head).transpose(1, 2)
        # k = [batch_size, n_heads, seq_len_K, hid_dim // n_heads]
        # q = [batch_size, n_heads, seq_len_q, hid_dim // n_heads]
        # attM = [batch_size, n_heads, seq_len_q, seq_len_K]
        return self.attM(q, k)

    def attM(self, query, key):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # ## focal part ## batch_size x key_len x query_len
        # funcF = scores * torch.sum(scores > 0, dim=-1, keepdim=True) - torch.sum(scores, dim=-1, keepdim=True)
        # focal_att = torch.where(funcF > 0, torch.ones_like(scores), torch.zeros_like(scores))
        # scores = scores * focal_att

        return scores


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, in_size,mid_size,out_size,dropout_r=0):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_size, mid_size)
        self.linear2 = nn.Linear(mid_size, out_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x=self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, hid_dim, dropout, max_len=2000):
        """
        three parameters:
        hid_dim：sequence_dim
        dropout: dropout rate
        max_len：the max len of the sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hid_dim)  # [max_len,hid_dim]
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0., hid_dim, 2) *
                             -(math.log(10000.0) / hid_dim))  # [1,hid_dim/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,max_len,hid_dim]
        self.register_buffer('pe', pe)  # regist buffer, not update parameters

    def forward(self, x):  # x = [1,wordnum,hid_dim]
        # x+position   x.size(1) is the sequence_len
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)
        return x


