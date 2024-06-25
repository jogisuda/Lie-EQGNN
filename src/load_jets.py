import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.sparse import coo_matrix

def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    # print(edge_mask[0])
    # raise
    for batch_idx in range(batch_size):
        nn = batch_idx*n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges

def collate_fn(data):
    data = list(zip(*data)) # label p4s nodes atom_mask
    data = [torch.stack(item) for item in data]
    batch_size, n_nodes, _ = data[1].size()
    atom_mask = data[-1]
    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    # diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    # edge_mask *= diag_mask

    edge_mask = data[-2]

    edges = get_adj_matrix(n_nodes, batch_size, edge_mask)
    return data + [edges]


def load_jets(p4s, nodes, labels, atom_mask, edge_mask, edges, batch_size, train_ratio, val_ratio, test_ratio):
    # p4s = torch.load('Roy/data/p4s.pt')
    # nodes = torch.load('Roy/data/nodes.pt')
    # labels = torch.load('Roy/data/labels.pt')
    # atom_mask = torch.load('Roy/data/atom_mask.pt')
    # edge_mask = torch.from_numpy(np.load('Roy/data/edge_mask.npy'))
    # edges = torch.from_numpy(np.load('Roy/data/edges.npy'))
    
    
    # Create a TensorDataset
    dataset_all = TensorDataset(labels, p4s, nodes, atom_mask, edge_mask)
    
    # Define the split ratios
    # train_ratio = 0.8
    # val_ratio = 0.1
    # test_ratio = 0.1
    
    # Calculate the lengths for each split
    total_size = len(dataset_all)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensure all data is used
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset_all, [train_size, val_size, test_size])
    
    # Create a dictionary to hold the datasets
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }
    
    # datasets = {split: TensorDataset(labels, p4s,
                                     # nodes, atom_mask, edge_mask) for split in ["train", "val", "test"]}
    
    
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     # sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
                                     pin_memory=False,
                                     # persistent_workers=True,
                                     collate_fn = collate_fn,
                                     drop_last=True if (split == 'train') else False,
                                     num_workers=0)
                        for split, dataset in datasets.items()}

    return dataloaders