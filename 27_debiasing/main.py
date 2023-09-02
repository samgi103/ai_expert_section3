import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed
from tensorboardX import SummaryWriter
from arguments import get_args
import time
import os 
args = get_args()


def main():

    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    result_dir = os.path.join(args.result_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(result_dir)
    writer = None
    if args.record:
        log_dir = os.path.join(args.log_dir, args.date, dataset, args.method)
        check_log_dir(log_dir)
        writer = SummaryWriter(log_dir + '/' + log_name)

    print(log_name)    
    ########################## get dataloader ################################
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset, 
                                                        batch_size=args.batch_size,
                                                        seed=args.seed,
                                                        n_workers=args.n_workers,
                                                        balSampling=args.balSampling,
                                                        args=args
                                                        )
    n_classes, n_groups, train_loader, test_loader = tmp
    ########################## get model ##################################
    model = networks.ModelFactory.get_model(args.model, n_classes, 224,
                                            pretrained=args.pretrained, n_groups=n_groups)

    model.cuda('cuda:{}'.format(args.device))
    if args.pretrained:
        if args.modelpath is not None:
            model.load_state_dict(torch.load(args.modelpath))
        
    print('successfully call the model')
#     set_seed(seed)
    scheduler=None
    ########################## get trainer ##################################
    if 'AdamW' == args.optim:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                optimizer=optimizer, scheduler=scheduler)

    ####################### start training or evaluating ####################
    
    if args.mode == 'train':
        start_t = time.time()
        trainer_.train(train_loader, test_loader, args.epochs, writer=writer)
        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        trainer_.save_model(save_dir, log_name)
    
    else:
        print('Evaluation ----------------')
        model_to_load = args.modelpath
        trainer_.model.load_state_dict(torch.load(model_to_load))
        print('Trained model loaded successfully')

    if args.evalset == 'all':
        trainer_.compute_confusion_matix('train', train_loader.dataset.n_classes, train_loader, result_dir, log_name)
        trainer_.compute_confusion_matix('test', test_loader.dataset.n_classes, test_loader, result_dir, log_name)

    elif args.evalset == 'train':
        trainer_.compute_confusion_matix('train', train_loader.dataset.n_classes, train_loader, result_dir, log_name)
    else:
        trainer_.compute_confusion_matix('test', test_loader.dataset.n_classes, test_loader, result_dir, log_name)
    if writer is not None:
        writer.close()
    print('Done!')


if __name__ == '__main__':
    main()
    
    

            
            

