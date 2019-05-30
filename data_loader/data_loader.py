from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader
from .dataset import TextDataset, TextBertDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TextDataLoader(BaseDataLoader):
    """
    Text Data loader
    """
    dataset_fn = TextDataset

    def __init__(self, cfg, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        trsfm = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()
        ])
        if training: split_dir = 'train'
        else: split_dir = 'test'

        self.dataset = self.dataset_fn(cfg, data_dir, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TextBertDataLoader(TextDataLoader):
    """
    Text Bert Data loader
    """
    dataset_fn = TextBertDataset
