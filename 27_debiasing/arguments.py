import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--result-dir', default='./results/',
                        help='directory to save results (default: ./results/)')
    parser.add_argument('--log-dir', default='./logs/',
                        help='directory to save logs (default: ./logs/)')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--t-device', default=0, type=int, help='teacher cuda device number')
    
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--evalset', default='all', choices=['all', 'train', 'test'])

    parser.add_argument('--dataset', required=True, default='', choices=['waterbird'])
    parser.add_argument('--img-size', default=176, type=int, help='img size for preprocessing')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20200101', type=str, help='experiment date')
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'fairdro','gdro'])

    parser.add_argument('--optim', default='Adam', type=str, required=False,
                        choices=['AdamP', 'AdamW','SGD', 'SGD_momentum_decay', 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--model', default='', required=True, choices=['resnet12', 'resnet50','cifar_net', 'resnet34', 'resnet18', 'resnet101','mlp', 'resnet18_dropout', 'bert','lr'])

    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--n-workers', default=1, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')

    parser.add_argument('--balSampling', default=False, action='store_true', help='balSampling loader')
    parser.add_argument('--record', default=False, action='store_true', help='record')
    
    # For lgdro chi,
    parser.add_argument('--rho', default=0.5, type=float, help='uncertainty box length')
    parser.add_argument('--use-01loss', default=False, action='store_true', help='using 0-1 loss when updating q')
    parser.add_argument('--gamma', default=0.1, type=float, help='learning rate for q')
    parser.add_argument('--optim-q', default='pd', choices=['pd', 'ibr', 'smt_ibr'], help='the type of optimization for q')
    parser.add_argument('--q-decay', default='linear', type=str, help='the type of optimization for q')
   
    args = parser.parse_args()
    args.cuda=True

    return args
