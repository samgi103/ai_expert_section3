import torch.nn as nn

from networks.resnet import resnet10, resnet12,resnet18, resnet34, resnet50, resnet101

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, n_classes=2, img_size=224, pretrained=False, n_groups=2):



        if 'resnet' in target_model:
            model_class = eval(target_model)
            if pretrained:
                model = model_class(pretrained=True, img_size=img_size)
                model.fc = nn.Linear(in_features=model.fc.weight.shape[1], out_features=n_classes, bias=True)
            else:
                model = model_class(pretrained=False, n_classes=n_classes, n_groups=n_groups, img_size=img_size)
            return model

        else:
            raise NotImplementedError


