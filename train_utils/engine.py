from matplotlib import rcParams
import json
import cv2
from itertools import chain
import os
import pandas as pd
import random
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vit_pytorch.efficient import ViT
from sklearn.utils import class_weight
import time
import copy
import timm

from pathlib import Path

from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn import metrics


def data_samplers(data, data_class, batch_size=None):
    '''retrurns data loaders called during training'''
    test_ids = [idx for idx in data['training']][:20]
    # a small subset for debugging if needed <^_^>
    data_tester = {key: data['training'][key] for key in test_ids}
    # change back

    train_data_loader = ava_data_reflect(
        data['training'], transform=reflect_transforms['training']
    )
    val_data_loader = ava_data_reflect(
        data['validation'], transform=reflect_transforms['validation']
    )
    test_data_loader = ava_data_reflect(
        data['test'], transform=reflect_transforms['test']
    )
    data_load_dict = {
        'training': train_data_loader,
        'validation': val_data_loader,
        'test': test_data_loader
    }
    # Let there be 9 samples and 1 sample in class 0 and 1 respectively
    labels = [data['training'][idx]['threshold'] for idx in data['training']]
    class_counts = np.bincount(labels)
    num_samples = sum(class_counts)
    # corresponding labels of samples
    class_weights = [num_samples/class_counts[i]
                     for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(weights), int(num_samples)
    )
    print(len(weights))
    print(class_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(weights), int(len(data['training'].keys()))
    )
    # with data sampler (note ->> must be same len[-,...,-] as train set!!)
    train_loader = DataLoader(
        dataset=train_data_loader,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset=val_data_loader,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_data_loader,
        batch_size=batch_size, shuffle=True)
    return {'training': train_loader, 'validation': val_loader, 'test': test_loader}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, criterion,
                optimizer,
                scheduler,
                num_epochs=None,
                model_name=None,
                did=None):
    '''Training and validation loops- 1 loop == one epoch
    has a saving fuciton saving model on best epoch
    records total train time'''
    results = {}
    # pathlib path object --> most pythonic option.
    did = Path(did)
    did = did/model_name
    os.makedirs(did, exist_ok=True)
    print(f'currently trianing {model_name}')
    print(f'{model_name} will be saved at {did/model_name}')
    since = time.time()
    # copy state dict for best model saving (training could make them worse)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            dataset_size = len(data[phase])
            if phase == 'training':
                model.train()   # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase)
            # Iterate over data.
            for inputs, labels, fid in tqdm(data_load_dict[phase], colour=('#FF69B4')):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'training':
                scheduler[0].step(scheduler[1])

            ballance = np.array([])
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            class_preds = outputs.argmax(dim=1)
            batch_acc = metrics.balanced_accuracy_score(labels.cpu(),
                                                        class_preds.cpu())
            ballance = np.append(ballance, batch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            key = 'epoch_'+str(epoch+1)+'_'+phase
            results[key] = {
                phase+' loss': epoch_loss,
                phase+' acc': float(epoch_acc.cpu()),
                phase + ' ballance_acc': ballance.mean()
            }
            print(results)
            # w mode to overwirte existing json - reading and re writing
            # in append modes can cause jsaon formatting issues
            # files are json for ease of loading to python dict in evaluation
            with open(did/(model_name+'.json'), 'w') as handle:
                json.dump(results, handle)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, did/model_name)
                print(f'Saving {model_name} in {did.name}')
                # model save

    time_elapsed = time.time() - since
    t_mins, t_seconds = time_elapsed // 60, time_elapsed % 60
    train_overall = {
        'mins': t_mins,
        'seconds': t_seconds,
        'best_acc': best_acc
    }
    print(f'training time = {train_overall}')
    with open(did/('train_overall'+'.json'), 'w') as handle:
        json.dump(train_overall, handle)


def loader(models):
    '''genrator for models when loooping through all models'''
    for mod in models:
        print(mod)
        if 'resnet' in mod:
            loaded = torch.load(model_locations[mod])
            feature_in = models[mod].fc.in_features
            models[mod].fc = nn.Linear(feature_in, 2)
            models[mod].load_state_dict(loaded['model_state_dict'])
            yield models[mod], mod
        elif'convit' in mod:
            loaded = torch.load(model_locations[mod])
            model = timm.create_model(mod, pretrained=True)
            model.head = nn.Linear(model.head.in_features, 2, bias=True)
            model.load_state_dict(loaded['model'])
            yield model, mod
        else:
            model = timm.create_model(mod, pretrained=True)
            model.head = nn.Linear(model.head.in_features, 2, bias=True)
            yield model, mod


def deep_eval(model):
    '''validatioan loop ruturns metrics dict for passed model'''
    color = colors()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    results_dict = {}
    with torch.no_grad():
        model.eval()
        for data, label, fid in tqdm(test_loader, colour=next(color)):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(output)
            for dir_, prob, lab in zip(fid, probabilities, label):
                results_dict[dir_.split('/')[-1]] = {
                    'class_probs': prob.cpu().tolist(),
                    'pred_class': int(prob.argmax(dim=0).cpu()),
                    'g_t_class': int(lab.cpu())}
            val_loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            results_dict['test_accuracy'] = {'test_acc': float(acc.cpu())}
    return results_dict
