
import importlib
import torch.utils.data as data
import numpy as np
from collections import defaultdict

dataset_dict = {
                'waterbird' : ['data_handler.waterbird', 'WaterBird'],
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, split='train', seed=0, balSampling=False, bs=256, method=None):
        root = f'./data/{name}' 
        kwargs = {'root':root,
                  'split':split,
                  'seed':seed,
                  }
         
        if name not in dataset_dict.keys():
            raise Exception('Not allowed method')
                
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        
        return class_(**kwargs)

class GenericDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, seed=0):
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.n_data = None
        self.y_array = None
        self.g_array = None
        
    def __len__(self):
        return np.sum(self.n_data)
    
    def _data_count(self, features, n_groups, n_classes):
        idxs_per_group = defaultdict(lambda: [])
        data_count = np.zeros((n_groups, n_classes), dtype=int)
    
        for idx, i in enumerate(features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            idxs_per_group[(s,l)].append(idx)
            
        print(f'mode : {self.split}')        
        for i in range(n_groups):
            print('# of %d group data : '%i, data_count[i, :])
        return data_count, idxs_per_group
            
    def _make_data(self, features, n_groups, n_classes):
        # if the original dataset not is divided into train / test set, this function is used
        import copy
        min_cnt = 100
        data_count = np.zeros((n_groups, n_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)
        
        train_data = features
        test_data = tmp
        return train_data, test_data

    def make_weights(self):
        '''
        Hint
        Utilize self.n_data, self.g_array, self.y_array
        self.n_data is a matrix where M_ij is the number of samples for i group and j class.
        self.g_array and self.y_array are lists where each entry is group or class label for each sample.  
        '''
        group_weights = len(self) / self.n_data
        weights = [group_weights[g,l] for g,l in zip(self.g_array,self.y_array)]
        return weights 