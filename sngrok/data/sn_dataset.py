import polars as pl
from sngrok.permutations import make_permutation_dataset
import torch
from torch.utils.data import Dataset



class SnDataset(Dataset):

    def __init__(self, n: int, mult_table: pl.DataFrame = None):
        self.n = n
        if mult_table is None:
            _, mult_table = make_permutation_dataset(n)
        self.data = mult_table.select(
            [pl.col('^index.*$'), pl.col('^conjugacy.*$')]
        )
        self.index_cols = ['index_left', 'index_right', 'index_target']
        self.conj_cols = ['conjugacy_class_left', 'conjugacy_class_right', 'conjugacy_class_right']

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        lperm, rperm, target = self.data.select(self.index_cols).row(index)
        conj_classes = self.data.select(self.conj_cols).row(index)
        conj_classes = tuple([str(conj) for conj in conj_classes])
        return (
            conj_classes,
            torch.tensor(lperm),
            torch.tensor(rperm),
            torch.tensor(target)
        )
