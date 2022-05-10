from torch.utils.data import DataLoader
from softmasked_bert.data.datasets.csc import CscDataset

def get_csc_loader(fp, _collate_fn, **kwargs):
    dataset = CscDataset(fp)
    loader = DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
    return loader
