import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, y, mol_vec = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y), torch.tensor(np.array(mol_vec))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path,exist_ok=True)


def integer_label_protein(sequence, max_length=1200, modelT5=False): #1200
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    if modelT5==True:
        encoding = np.float32(np.zeros([max_length, 1024]))
        seqlen = min(sequence.shape[0], max_length)
        encoding[:seqlen, :] = sequence[:seqlen, :]
    else:
        encoding = np.zeros(max_length)
        for idx, letter in enumerate(sequence[:max_length]):
            try:
                letter = letter.upper()
                encoding[idx] = CHARPROTSET[letter]
            except KeyError:
                logging.warning(
                    f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
                )
        seqlen=idx+1
    return encoding, seqlen
