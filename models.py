import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ban import FocalAtt, SCAtt, SorCAtt, LayerNorm
from torch.nn.utils.weight_norm import weight_norm

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, device, config):
        super(DrugBAN, self).__init__()

        self.drug_extractor = MolecularGCN(config)
        self.d_dim=config.DRUG.HIDDEN_LAYERS[-1]
        self.p_dim=config.PROTEIN.NUM_FILTERS[-1]
        self.modelT5=config.modelT5
        self.noCNN=config.noCNN
        self.modelmol=config.modelmol
        self.nofocal=config.nofocal
        self.Fe1=config.Fe1
        self.Fe2=config.Fe2
        self.noPE=config.noPE
        self.per=config.per

        if config.modelT5 and config.noCNN:
            self.p_dim = 1024
        else:
            self.protein_extractor = ProteinCNN(config, config.modelT5)
        self.modelmol=config.modelmol
        if self.modelmol:
            self.FC_d=nn.Linear(300, self.d_dim, bias=True)
            self.d_dim=self.d_dim+self.d_dim

        self.FocalAtt= FocalAtt(config, self.d_dim, self.p_dim, mode="CA")

        self.CAtt_pro1 = SorCAtt(config,config.n_layerC,self.p_dim,self.d_dim)
        self.CAtt_pro2 = SorCAtt(config,config.n_layerC,self.p_dim,config.hid_dim)

        self.adapted_w = nn.Parameter(torch.ones(2, config.hid_dim))

        self.mlp_classifier = MLPDecoder(config)


    def forward(self, bg_d, v_p, mol_vec, mode="train"):
        v_d = self.drug_extractor(bg_d)
        if not (self.modelT5 and self.noCNN):
            v_p = self.protein_extractor(v_p)

        if self.modelmol:
            mol_vec=self.FC_d(mol_vec)
            v_d = torch.cat([v_d,mol_vec],-1)

        softmax = True
        nofocal=self.nofocal
        Fe1= self.Fe1
        Fe2= self.Fe2   # Fe2=True的优先级 高于 Fe1=True
        per=self.per #抽取蛋白质氨基酸的比例
        if self.noPE:
            PE = ''
        else:
            PE='qk' #位置编码

        att1=None
        att2=None
        att3=None
        attidx_d=None
        attidx_p=None

        if not Fe2:
            _, att1, feature1 = self.CAtt_pro1(v_p, v_d, SF=softmax, PE=PE, outmode="bilinear")

        if Fe1 and not Fe2:
            feature=feature1
        else:
            # focal att  选择是否更新v_d
            _, v_d2, v_p2, att2, attidx_d, attidx_p = self.FocalAtt(v_d, v_p, per, SF = softmax, PE = PE, outmode="attedfocal")

            if nofocal:
                # nofocal
                _, att3, feature2 = self.CAtt_pro2(v_p, v_d2, SF=softmax, PE=PE, outmode="bilinear")
            else:
                _, att3, feature2 = self.CAtt_pro2(v_p2, v_d2, SF=softmax, PE=PE, outmode="bilinear")
            if Fe2:
                feature=feature2
            else:
                feature = torch.cat([feature1.unsqueeze(1), feature2.unsqueeze(1)], 1)
                adapted_w = torch.softmax(self.adapted_w, 0)
                feature = torch.mul(feature, adapted_w.unsqueeze(0)).sum(1)

        score = self.mlp_classifier(feature)

        if mode == "train":
            return score
        else:
            return score, att1, att2, att3, attidx_d, attidx_p


class MolecularGCN(nn.Module):
    def __init__(self, __C):
        super(MolecularGCN, self).__init__()
        in_feats=__C.DRUG.NODE_IN_FEATS
        dim_embedding = __C.DRUG.NODE_IN_EMBEDDING
        padding = __C.DRUG.PADDING
        hidden_feats = __C.DRUG.HIDDEN_LAYERS
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=None) # activation=None 默认为relu
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats



class ProteinCNN(nn.Module):
    def __init__(self, __C,modelT5):
        super(ProteinCNN, self).__init__()
        embedding_dim=__C.PROTEIN.EMBEDDING_DIM
        num_filters=__C.PROTEIN.NUM_FILTERS
        kernel_size=__C.PROTEIN.KERNEL_SIZE
        padding=__C.PROTEIN.PADDING
        self.modelT5=modelT5
        if self.modelT5==True:
            self.embedding = nn.Linear(1024, embedding_dim)
        else:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        self.nLayers = len(num_filters)
        in_ch = [embedding_dim] + num_filters
        self.cnnLayers = nn.ModuleList([])
        for idx in range(self.nLayers):
            if padding:
                pad=[kernel_size[idx] // 2, kernel_size[idx] // 2-1+kernel_size[idx] % 2,0,0]
            else:
                pad=[0,0,0,0]
            self.cnnLayers.append(nn.Sequential(
                nn.ZeroPad2d(pad),
                nn.Conv1d(in_channels=in_ch[idx], out_channels=in_ch[idx + 1], kernel_size=kernel_size[idx], padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(in_ch[idx + 1])
            ))

    def forward(self, v):
        if self.modelT5 == True:
            v = self.embedding(v)
        else:
            v = self.embedding(v.long())
        for idx in range(self.nLayers):
            v = self.cnnLayers[idx](v.transpose(2, 1)).transpose(2, 1)
        return v

class MLPDecoder(nn.Module):
    def __init__(self, __C):
        super(MLPDecoder, self).__init__()
        in_dim = __C.MLP.IN_DIM
        hidden_dim = __C.MLP.HIDDEN_DIM
        out_dim = __C.MLP.OUT_DIM
        out_binary = __C.MLP.BINARY

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.fc3 = nn.Linear(out_dim, out_binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.hidden_size = __C.flat_hidden_size
        self.mlp = FFN(__C.flat_in_size, __C.flat_mid_size, __C.flat_hidden_size, __C.dropout)
        self.linear_merge = nn.Linear(__C.flat_in_size * __C.flat_hidden_size, __C.flat_out_size)

    def forward(self, x):
        att = self.mlp(x)
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.hidden_size):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted

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
