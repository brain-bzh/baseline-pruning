import torch
import torchvision
import os
import io

__all__ = ['abcpruner30', 'abcpruner50', 'abcpruner70', 'abcpruner80', 'abcpruner100']

content = {'ABCPruner30': 'res50_top1_70.289_top5_89.631',
           'ABCPruner50': 'res50_top1_72.582_top5_90.919',
           'ABCPruner70': 'res50_top1_73.516_top5_91.512',
           'ABCPruner80': 'res50_top1_73.864_top5_91.687',
           'ABCPruner100': 'res50_top1_74.843_top5_92.272',
           }


def cat_files_in_folder(path):
    listdir = sorted(os.listdir(path))

    data = b''
    for filename in listdir:
        with open(os.path.join(path, filename), 'rb') as f:
            data += f.read()

    return io.BytesIO(data)


def load_model(path):
    net = torchvision.models.resnet50(weights=None)

    file = cat_files_in_folder(path)
    state_dict = torch.load(file)
    for n, m in net.named_modules():
        if hasattr(m, 'weight'):
            param_name = n + '.weight'
            if param_name in state_dict['state_dict']:
                m.weight.data = state_dict['state_dict'][param_name]
        if hasattr(m, 'bias'):
            param_name = n + '.bias'
            if param_name in state_dict['state_dict']:
                m.bias.data = state_dict['state_dict'][param_name]
        if hasattr(m, 'running_mean'):
            param_name = n + '.running_mean'
            if param_name in state_dict['state_dict']:
                m.running_mean.data = state_dict['state_dict'][param_name]
        if hasattr(m, 'running_var'):
            param_name = n + '.running_var'
            if param_name in state_dict['state_dict']:
                m.running_var.data = state_dict['state_dict'][param_name]
    return net


def abcpruner30():
    path = os.path.join(os.path.dirname(__file__), content['ABCPruner30'])
    return load_model(path)


def abcpruner50():
    path = os.path.join(os.path.dirname(__file__), content['ABCPruner50'])
    return load_model(path)


def abcpruner70():
    path = os.path.join(os.path.dirname(__file__), content['ABCPruner70'])
    return load_model(path)


def abcpruner80():
    path = os.path.join(os.path.dirname(__file__), content['ABCPruner80'])
    return load_model(path)


def abcpruner100():
    path = os.path.join(os.path.dirname(__file__), content['ABCPruner100'])
    return load_model(path)
