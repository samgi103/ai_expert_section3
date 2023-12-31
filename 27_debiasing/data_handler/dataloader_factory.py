from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, batch_size=256, seed = 0, n_workers=4,
                       balSampling=False, args=None):

        test_dataset = DatasetFactory.get_dataset(name, split='test',
                                                  seed=seed, bs=batch_size,method=args.method)
        train_dataset = DatasetFactory.get_dataset(name, split='train',
                                                   seed=seed, bs=batch_size,method=args.method)
        
        n_classes = test_dataset.n_classes
        n_groups = test_dataset.n_groups
        
        def _init_fn(worker_id):
            np.random.seed(int(seed))

        shuffle = True
        sampler = None
        if balSampling:
#             if args.method == 'mfd':
#                 from data_handler.custom_loader import Customsampler                
#                 sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
#             else:
            from torch.utils.data.sampler import WeightedRandomSampler
            weights = train_dataset.make_weights(args.method)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#             else:
#                 from data_handler.custom_loader import Customsampler                
#                 sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
            shuffle = False

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      num_workers=n_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)

        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                     num_workers=n_workers, worker_init_fn=_init_fn, pin_memory=True)

        print('# of test data : {}'.format(len(test_dataset)))
        print('# of train data : {}'.format(len(train_dataset)))
        print('Dataset loaded.')
        print('# of classes, # of groups : {}, {}'.format(n_classes, n_groups))

        return n_classes, n_groups, train_dataloader, test_dataloader

