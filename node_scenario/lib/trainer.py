import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_latest_file, iterate_minibatches, check_numpy, process_in_chunks
from .nn_utils import to_one_hot
from collections import OrderedDict
from copy import deepcopy
from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, auc, roc_curve
from sklearn import preprocessing
from itertools import cycle

import matplotlib.pyplot as plt

import pickle

class Trainer(nn.Module):
    def __init__(self, model, loss_function, experiment_name=None, warm_start=False, 
                 Optimizer=torch.optim.Adam, optimizer_params={}, verbose=False, 
                 n_last_checkpoints=1, **kwargs):
        """
        :type model: torch.nn.Module
        :param loss_function: the metric to use in trainnig
        :param experiment_name: a path where all logs and checkpoints are saved
        :param warm_start: when set to True, loads last checpoint
        :param Optimizer: function(parameters) -> optimizer
        :param verbose: when set to True, produces logging information
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.verbose = verbose
        self.opt = Optimizer(list(self.model.parameters()), **optimizer_params)
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)

        self.experiment_path = os.path.join('./logs/', experiment_name)
        if not warm_start and experiment_name != 'debug':
            assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)
        if warm_start:
            self.load_checkpoint()
    
    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = "temp_{}".format(self.step)
        if path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step)
        ]), path)
        # if self.verbose:
        #     print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        elif tag is not None and path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])

        # if self.verbose:
        #     print('Loaded ' + path)
        return self

    def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None):
        assert tags is None or paths is None, "please provide either tags or paths or nothing, not both"
        assert out_tag is not None or out_path is not None, "please provide either out_tag or out_path or both, not nothing"
        if tags is None and paths is None:
            paths = self.get_latest_checkpoints(
                os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.n_last_checkpoints)
        elif tags is not None and paths is None:
            paths = [os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

        checkpoints = [torch.load(path) for path in paths]
        averaged_ckpt = deepcopy(checkpoints[0])
        for key in averaged_ckpt['model']:
            values = [ckpt['model'][key] for ckpt in checkpoints]
            averaged_ckpt['model'][key] = sum(values) // len(values)

        if out_path is None:
            out_path = os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
        torch.save(averaged_ckpt, out_path)

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        assert len(list_of_files) > 0, "No files found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.n_last_checkpoints
        paths = self.get_latest_checkpoints(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            os.remove(ckpt)

    def train_on_batch(self, *batch, device):
        x_batch, y_batch = batch
        x_batch = torch.as_tensor(x_batch, device=device)
        y_batch = torch.as_tensor(y_batch, device=device)
        # print("x_batch:\n", x_batch[0:5])
        # print("y_batch:\n", y_batch[0:5])
        # print("x_batch.shape: ", x_batch.shape)
        # print("y_batch.shape: ", y_batch.shape)
        self.model.train()
        self.opt.zero_grad()
        # print('out_shape:\n', self.model(x_batch))
        loss = self.loss_function(self.model(x_batch), y_batch).mean()
        # print('loss: ', loss)
        loss.backward()
        self.opt.step()
        self.step += 1
        self.writer.add_scalar('train loss', loss.item(), self.step)
        
        return {'loss': loss}

    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)
            error_rate = (y_test != np.argmax(logits, axis=1)).mean()
        return error_rate

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()
        return error_rate
    
    def evaluate_auc(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            # print("logits:\n", process_in_chunks(self.model, X_test, batch_size=batch_size))
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            auc = roc_auc_score(check_numpy(to_one_hot(y_test)), logits, average="macro", multi_class="ovo")
        return auc
    
    def evaluate_logloss(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            logloss = log_loss(check_numpy(to_one_hot(y_test)), logits)
        return logloss

    def evaluate_precision(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        # print("y_test:\n", y_test)
        with torch.no_grad():
            # print("logits:\n", process_in_chunks(self.model, X_test, batch_size=batch_size))
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_pre = np.argmax(logits, axis=1)
            y_test = torch.tensor(y_test)
            pre = precision_score(y_test, y_pre, labels=[0,1,2,3], zero_division=1, average='macro')
        return pre

    def evaluate_recall(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        # print("y_test:\n", y_test)
        with torch.no_grad():
            # print("logits:\n", process_in_chunks(self.model, X_test, batch_size=batch_size))
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_pre = np.argmax(logits, axis=1)
            y_test = torch.tensor(y_test)
            recall = recall_score(y_test, y_pre, labels=[0,1,2,3], average='macro', zero_division=1)
        return recall

    def draw_roc_curve(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)

        with torch.no_grad():
            # print("logits:\n", process_in_chunks(self.model, X_test, batch_size=batch_size))
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            
        y_true = y_test
        y_predict = logits
        
        y = preprocessing.label_binarize(y_true, classes=[0, 1, 2, 3])
        n_classes = y.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            # 计算每一类的ROC
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        lw = 2
        plt.figure()

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average Si-Ch ROC curve (area = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        
        pickle.dump(fpr, open('./roc/Sh2Ch_fpr.txt', 'wb'))
        pickle.dump(tpr, open('./roc/Sh2Ch_tpr.txt', 'wb'))

        # getback = pickle.load(open('C:/Users/sliu/Desktop/tmp.txt', 'rb'))



        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #              label='ROC curve of class {0} (area = {1:0.4f})'
        #                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of Model')
        plt.legend(loc="lower right")
        plt.show()