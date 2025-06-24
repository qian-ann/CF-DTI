from yacs.config import CfgNode as CN

_C = CN()

_C.dropout=0.2
_C.hid_dim=256
_C.n_heads=8 #8
_C.mid_size=_C.hid_dim*2
_C.out_size=_C.hid_dim
_C.n_layerS=0 #CoAtt层数 Self
_C.n_layerC=1 #CoAtt层数 Cross
_C.flat_in_size=_C.hid_dim
_C.flat_mid_size=_C.hid_dim*2
_C.flat_hidden_size=4   #AttFlat混合用
_C.flat_out_size=_C.hid_dim
_C.modelT5=False   #是否使用T5
_C.noCNN=False  #是否不使用CNN对蛋白质特征进行特征提取
_C.modelmol=False  #是否使用 mol2vec的特征

_C.nofocal=False  #是否使用焦点注意力
_C.Fe1=False  #是否只使用粗粒度
_C.Fe2=False #是否只使用细粒度
_C.noPE=False  #是否不适用位置编码
_C.per=0.1 # 0.1 #抽取蛋白质氨基酸的比例




# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# MLP decoder
_C.MLP = CN()
_C.MLP.NAME = "MLP"
_C.MLP.IN_DIM = _C.hid_dim
_C.MLP.HIDDEN_DIM = _C.hid_dim*2
_C.MLP.OUT_DIM = 128
_C.MLP.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.LR = 5e-5 # 1e-3 #
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048
_C.SOLVER.SEEDMODE=0

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

def get_cfg_defaults():
    return _C.clone()
