import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class FollowSuit(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with image indexes and class label annotations.
        """
        self.data = list()
        self.follow_suit = pd.read_csv(csv_file).values
        self.classes = [1, 2, 3, 4]

    def __len__(self):
        return len(self.follow_suit)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.follow_suit[idx]
        X = row[:-1]
        y = row[len(row)-1]
        return X, y


def load_follow_suit_data(base_dir='.', train_batch_size=1, test_batch_size=1, size='small', repeats=[1,2,3,4,5]):
    train_loads = []
    test_ds = FollowSuit(base_dir + '/data/unstructured_data/'+size+'/test.csv')
    test_load = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
    for split in repeats:
        train_ds = FollowSuit(base_dir+'/data/unstructured_data/'+size+'/train_{0}.csv'.format(str(split)))
        train_loads.append(torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size))
    return train_loads, test_load
