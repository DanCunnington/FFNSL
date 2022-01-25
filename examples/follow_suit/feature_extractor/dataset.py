import torch
import pandas as pd
import os
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset
from torchvision.transforms import transforms

suits = ['h', 'c', 's', 'd']
cards = ['a', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k']


class PlayingCards(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with label annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = list()
        self.root_dir = root_dir
        self.playing_cards = pd.read_csv(csv_file)
        self.transform = transform
        self.mapping = []

        for s in suits:
            for c in cards:
                self.mapping.append(c+s)

    def __len__(self):
        return len(self.playing_cards)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.playing_cards.iloc[idx, 0])
        image = io.imread(img_name)
        str_label = self.playing_cards.iloc[idx, 1]
        label = self.mapping.index(str_label)

        if self.transform:
            image = self.transform(image)

        return image.float(), label


def load_data(root_dir='.', deck='standard', train_batch_size=32, test_batch_size=32):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((274, 174)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_loader = None
    if deck == 'standard':
        train_dataset = PlayingCards(root_dir+'/data/'+deck+'/train.csv', root_dir+'/data/'+deck+'/imgs', transform=t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)

    test_dataset = PlayingCards(root_dir+'/data/'+deck+'/test.csv', root_dir+'/data/'+deck+'/imgs', transform=t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, test_loader



