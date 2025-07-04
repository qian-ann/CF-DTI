import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein



class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, dataT5, datamol, max_drug_nodes=290, max_protein_length=1200, modelT5=False, modelmol=False):
        self.list_IDs = list_IDs
        self.df = df
        self.dataT5 = dataT5
        self.datamol = datamol
        self.modelT5=modelT5
        self.modelmol = modelmol
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_length = max_protein_length

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats

        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        if self.modelT5==True:
            aa = 'a' + str(index)
            v_p, protein_len = integer_label_protein(self.dataT5[aa][()],self.max_protein_length,True)
        else:
            v_p = self.df.iloc[index]['Protein']
            v_p, protein_len= integer_label_protein(v_p)

        if self.modelmol==True:
            aa = 'a' + str(index)
            mol_vec = self.datamol[aa][()]
            mol_vec = np.float32(np.concatenate([mol_vec,np.zeros([self.max_drug_nodes -mol_vec.shape[0],300])],0))
        else:
            mol_vec = np.zeros([1,1])

        y = self.df.iloc[index]["Y"]
        return v_d, v_p, y, mol_vec


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
