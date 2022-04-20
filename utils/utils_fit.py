import os

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


class BiasLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, normalisation_mode='global'):
        super(BiasLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.norm_mode = normalisation_mode
        self.global_min = 100000

    def norm_global(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        if min < self.global_min:
            self.global_min = min
        normalised = ((tensor - self.global_min) / (max - min))
        return normalised

    def norm_local(self, tensor):
        min = tensor.clone().min()
        max = tensor.clone().max()

        normalised = ((tensor - min) / (max - min))

        return normalised

    def forward(self, features, output, target):
        features_copy = features.clone().detach()
        features_dp = features_copy.reshape(features_copy.shape[0], -1)

        features_dp = (torch.var(features_dp, dim=1))
        if self.norm_mode == 'global':
            variance_dp_normalised = self.norm_global(features_dp)
        else:
            variance_dp_normalised = self.norm_local(features_dp)

        weights = ((torch.exp(variance_dp_normalised * self.beta) - 1.) / 1.) + self.alpha
        loss = weights * self.ce(output, target)

        loss = loss.mean()

        return loss


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir):
    total_loss      = 0
    total_accuracy  = 0
    criterion = BiasLoss()
    val_loss        = 0
    val_accuracy    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            images, targets = batch
            with torch.no_grad():
                images      = torch.from_numpy(images).type(torch.FloatTensor)
                targets     = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()
            outputs     = model_train(images)
            loss_value = criterion(features=images, output=outputs, target=targets)
            loss_value.backward()
            optimizer.step()
            # print(model.state_dict())
            total_loss += loss_value.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs     = model_train(images)
                loss_value = criterion(features=images, output=outputs, target=targets)
                
                val_loss    += loss_value.item()
                accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                val_accuracy    += accuracy.item()
                
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
  
