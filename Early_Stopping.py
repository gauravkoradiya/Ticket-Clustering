import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, PATH, patience=7, verbose=False):
        """
        Args:
            PATH : Location where u want to store your checkpoint
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.path = PATH
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf


    def __call__(self,model,train_loss,optimizer,epoch):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.path,train_loss,model,optimizer,epoch)

        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(self.path,train_loss, model,optimizer,epoch)
            self.counter = 0

    def save_checkpoint(self,Path,train_loss, model,optimizer,epoch):
        '''Saves model when validation loss decrease.'''
        SAVE_DIR = Path
        # #MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')
        if self.verbose:
            print('Train loss decreased({} -->{}).  Saving model ...'.format(self.train_loss_min,train_loss))
        torch.save({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
        }, os.path.join(SAVE_DIR ,'train_loss-{}_epoch-{}_checkpoint.pth.tar'.format(train_loss,epoch)))
        self.train_loss_min = train_loss