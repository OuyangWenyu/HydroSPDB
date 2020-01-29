import torch
from torch.utils.data import Dataset, DataLoader


class SimInputDataset(Dataset):
    """simulated streamflow input"""

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.data_flow = data_source.prepare_flow_data()
        self.data_target = data_source.read_outflow()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data_flow[index]
        y = self.data_target[index]
        return x, y

    def __len__(self):
        return len(self.data_flow)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape
            - trg_seq: torch tensor of shape
    Returns:
        src_seqs: torch tensor of shape (time_seq_length ,batch_size, one_unit_length).
        trg_seqs: torch tensor of shape (time_seq_length ,batch_size, one_unit_length).
    """

    def merge(sequences):
        padded_seqs = torch.zeros(sequences[0].shape[0], len(sequences), sequences[0].shape[1])
        for i, seq in enumerate(sequences):
            padded_seqs[:, i, :] = seq
        return padded_seqs

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # transform sequences (from 2D tensor to 3D tensor)
    src_seqs = merge(src_seqs)
    trg_seqs = merge(trg_seqs)
    return src_seqs, trg_seqs


def get_loader(dataset, batch_size=100, shuffle=False, num_workers=0):
    """Returns data loader for custom dataset.
    Args:
        dataset: dataset
        batch_size: mini-batch size.
        shuffle: is shuffle?
        num_workers: num of cpu core
    Returns:
        data_loader: data loader for custom dataset.
    """
    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    if num_workers < 1:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                 num_workers=num_workers)
    return data_loader
