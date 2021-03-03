#!/usr/bin/env python
# coding: utf-8


import time
import argparse
import torch, torch.nn as nn
import torch.nn.functional as F
from lib.data_process import *
import lib

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 256, help='batchsize')
parser.add_argument('--dataset', type=str, default='beijing',choices=["beijing","shanghai","sichuan", "illinois"], help='which dataset to use')
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment_name = 'ip_scenario'
experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}.{:0>2d}'.format(experiment_name, *time.gmtime()[:5])
print("experiment:", experiment_name)


# dataset choice ["beijing", "shanghai", "sichuan", "illinois"]
data = Dataset(opt.dataset, random_state=1334, quantile_transform=True)

in_features = data.X_train.shape[1]
num_classes = len(set(data.y_train))


data.X_train = torch.tensor(data.X_train, dtype=torch.float32)
data.X_valid = torch.tensor(data.X_valid, dtype=torch.float32)
data.X_test = torch.tensor(data.X_test, dtype=torch.float32)

data.y_train = data.y_train.astype('int64')
data.y_valid = data.y_valid.astype('int64') 
data.y_test = data.y_test.astype('int64') 


model = nn.Sequential(
    lib.DenseBlock(in_features, layer_dim=256, num_layers=4, tree_dim=num_classes+1, depth=6, flatten_output=False,
                   choice_function=lib.entmax15, bin_function=lib.entmoid15),
    lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),  # average first channels of every tree
    
).to(device)

with torch.no_grad():
    res = model(torch.as_tensor(data.X_train[:1000], device=device))
    
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


from qhoptim.pyt import QHAdam
optimizer_params = { 'nus': (0.7, 1.0), 'betas': (0.95, 0.998) }


trainer = lib.Trainer(
    model=model, loss_function=F.cross_entropy,
    experiment_name=experiment_name,
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    verbose=True,
    n_last_checkpoints=5
)


loss_history_step, auc_history_step = [], []
loss_history, mse_history, auc_history = [], [], []
best_auc = 0
best_step_auc = 0
early_stopping_rounds = 1000
report_frequency = 100


print("------ training starts ------")

for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=opt.batch_size,
                                                shuffle=True, epochs=float("inf")):
    metrics = trainer.train_on_batch(*batch, device=device)
    loss_history_step.append(metrics['loss'].cpu().detach().numpy())
    auc = trainer.evaluate_auc(data.X_valid, data.y_valid, device=device, batch_size=1024)
    auc_history_step.append(auc)
    if auc > best_auc:
            best_auc = auc
            best_step_auc = trainer.step
            trainer.save_checkpoint(tag='best_auc')
            

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')

        auc = trainer.evaluate_auc(
            data.X_valid, data.y_valid, device=device, batch_size=1024)

        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()
        print("Step: {}   Loss: {:.5f}    Val AUC: {:.5f}".format(trainer.step, metrics['loss'], auc))
    if trainer.step > best_step_auc + early_stopping_rounds:
        print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step_auc)
        print("Best Val AUC: %0.5f" % (best_auc))
        break

print("------ training ends ------")

trainer.load_checkpoint(tag='best_auc')
auc = trainer.evaluate_auc(data.X_test, data.y_test, device=device)
pre = trainer.evaluate_precision(data.X_test, data.y_test, device=device)
recall = trainer.evaluate_recall(data.X_test, data.y_test, device=device)
print('Best step: ', trainer.step)
print("Test precision: %0.5f" % (pre))
print("Test recall: %0.5f" % (recall))
print("Test auc: %0.5f" % (auc))

