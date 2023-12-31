import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import cvxpy as cvx
import time 

def chi_proj(pre_q, rho):
    #start = time.time()
    g = pre_q.shape[0]
    q = cvx.Variable(g)
    v = pre_q.cpu().numpy()
    #obj = cvx.Minimize(cvx.square(cvx.norm(q - v, 2)))
    obj = cvx.Minimize(cvx.sum(cvx.kl_div(q, v)))

    constraints = [q>= 0.0,
                   cvx.sum(q)==1.0,
                   cvx.square(cvx.norm(q-np.ones(g)/g, 2)) <= rho*2/g]
    
    prob = cvx.Problem(obj, constraints)
    prob.solve() # Returns the optimal value.
    print("optimal value : ", prob.value)
    print("pre q : ", pre_q)
    print("optimal var :", q.value)
    #end = time.time()
    #print(f'took {end-start} s')
    return q.value


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_accuracy(outputs, labels, binary=False, reduction='mean'):
    #if multi-label classification
    if len(labels.size())>1:
        outputs = (outputs>0.0).float()
        correct = ((outputs==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    
    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
        
    c = (predictions == labels).float().squeeze()
    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.item()

def get_subgroup_accuracy(outputs, labels, groups, n_classes, n_groups, reduction='mean'):
    n_subgroups = n_classes*n_groups
    with torch.no_grad():
        subgroups = groups * n_classes + labels
        group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_denom = group_denom.reshape((n_groups, n_classes))
       
        predictions = torch.argmax(outputs, 1)
        c = (predictions==labels).float()

        num_correct = (group_map @ c).reshape((n_groups, n_classes))
        subgroup_acc = num_correct/group_denom
        group_acc = num_correct.sum(1) / group_denom.sum(1) 
        
    return subgroup_acc,group_acc 
    
def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")

def make_log_name(args):
    log_name = args.model

    if args.pretrained:
        log_name += '_pretrained'

    log_name += f'_seed{args.seed}_epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}_{args.optim}'
    log_name += f'_wd{args.weight_decay}'

    if 'gdro' in args.method:
        log_name += f'_gamma{args.gamma}'

    elif args.method == 'fairdro':
        log_name += f'_{args.optim_q}'
        if args.optim_q == 'smt_ibr':
            log_name += f'_{args.q_decay}'
        log_name += f'_rho{args.rho}'
        if args.use_01loss:
            log_name +='_01loss'
        
    if args.balSampling:
        log_name += '_balSampling'


    return log_name
