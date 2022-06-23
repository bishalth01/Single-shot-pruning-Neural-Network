import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class DataPrep:
    def __init__(self, dataset, args) -> None:
        assert dataset in ['cifar10', 'cifar100'], "Dataset not implemented, check dataset name for misspell!"
        self.dataset = dataset
        self.args = args
        self.train_batch_size = args.batch_size
        self.test_batch_size = 128
        self.workers = args.workers

    def get_dataset(self):
        if self.dataset == 'cifar10':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize, ]))

        if self.dataset == 'cifar100':
            print("Preparing Cifar100 dataset!")
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2023, 0.1994, 0.2010])
            train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

            test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize, ]))
        return train_dataset, test_dataset

    def get_loaders(self):
        train_dataset, test_dataset = self.get_dataset()

        if self.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.train_batch_size, shuffle=True,
                                                       num_workers=self.workers, pin_memory=True)

            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.test_batch_size, shuffle=False,
                                                      num_workers=self.workers, pin_memory=True)

        if self.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.train_batch_size, shuffle=True,
                                                       num_workers=self.workers, pin_memory=True)

            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.test_batch_size, shuffle=False,
                                                      num_workers=self.workers, pin_memory=True)
        return train_loader, test_loader

    def _split_dataset(self, train_dataset, args):
        train_size_A = int(0.5 * len(train_dataset))
        train_size_B = len(train_dataset) - train_size_A
        trainset_A, trainset_B = torch.utils.data.random_split(train_dataset, [train_size_A, train_size_B])

        train_loader_A = torch.utils.data.DataLoader(trainset_A,
                                                     batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True)

        train_loader_B = torch.utils.data.DataLoader(trainset_B,
                                                     batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True)
        return train_loader_A, train_loader_B

    def get_split_dataloaders(self):
        train_dataset, test_dataset = self.get_dataset()
        _, test_loader = self.get_loaders()

        train_loader_A, train_loader_B = self._split_dataset(train_dataset, self.args)
        return train_loader_A, train_loader_B, test_loader

    def _multi_split_dataset(self, train_dataset, args, split_num):
        split_size = int((1 / split_num) * len(train_dataset))
        split_size_l = [split_size] * split_num
        split_trainsets = torch.utils.data.random_split(train_dataset, split_size_l)

        split_trainloaders = list()
        for id in range(split_num):
            split_trainloaders.append(torch.utils.data.DataLoader(split_trainsets[id],
                                                                  batch_size=args.batch_size, shuffle=True,
                                                                  num_workers=args.workers, pin_memory=True)
                                      )
        return split_trainloaders

    def get_multi_split_loaders(self, split_num=4):
        train_dataset, test_dataset = self.get_dataset()
        _, test_loader = self.get_loaders()

        split_trainloaders = self._multi_split_dataset(train_dataset, self.args, split_num)
        return split_trainloaders, test_loader