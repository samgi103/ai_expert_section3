from __future__ import print_function
from collections import defaultdict

import time
from utils import get_accuracy
import trainer
import torch
import numpy as np


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.gamma = args.gamma # learning rate of adv_probs
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        model = self.model
        model.train()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        
        self.adv_probs = torch.ones(n_groups*n_classes).cuda() / (n_groups*n_classes)
        
        for epoch in range(epochs):
            
            self._train_epoch(epoch, train_loader, model,criterion)

            eval_start_time = time.time()                
            eval_loss, eval_acc, eval_dcam, eval_dcaa, _, _  = self.evaluate(self.model, 
                                                                             test_loader, 
                                                                             self.criterion,
                                                                             epoch, 
                                                                             train=False,
                                                                             record=self.record,
                                                                             writer=writer
                                                                            )
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_dcam, (eval_end_time - eval_start_time)))

            if self.record:
                self.evaluate(self.model, train_loader, self.criterion, epoch, 
                              train=True, 
                              record=self.record,
                              writer=writer
                             )
                
            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        
        n_classes = train_loader.dataset.n_classes
        n_groups = train_loader.dataset.n_groups
        n_subgroups = n_classes * n_groups
        
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            subgroups = groups * n_classes + labels
            outputs = model(inputs)

            loss = self.train_criterion(outputs, labels)

            # calculate the groupwise losses
            group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom

            # update q
            self.adv_probs = self.adv_probs * torch.exp(self.gamma*group_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum()) # proj

            loss = group_loss @ self.adv_probs
                
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()




