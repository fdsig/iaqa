
import json 
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import json
import sklearn
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.feature_selection import r_regression


import json
import matplotlib.pylab as pl

def get_metrics(fid=None):
    '''Returns metrics dictionary, global maximum an minimum values
    takes file path as argument and gets txt files from all logs genrated
    by convit training'''
    result_fids = [fid for fid in os.scandir(fid) 
               if 'txt' in fid.name]

    results_dict =  { }
    for fid_ in result_fids:
        with open(fid_, 'r') as fid:
            text = fid.readlines()
            key = fid_.name.split('.')[0]
            results_dict[key]=text
    results = {
         i:{'epoch_'+str(json.loads(j)['epoch']):json.loads(j) for j in results_dict[i]} 
         for i in results_dict
     }
    sorted_keys = sorted(list(results.keys()))
    results = {key:results[key] for key in sorted_keys}
    deep_keys =  [list(results[i][j].keys() )for i in results for j in list(results[i].keys())][0]
    metrics = {deep_keys[k]:{i:np.array([results[i][j][deep_keys[k]] 
               for j in results[i]][:10]) 
            for i in results} for k in range(1,4)}
    

    train_loss, test_loss, test_acc = (metrics[i] for i in metrics)
    test_acc = {key:test_acc[key]/100 for key in test_acc}
    metrics['test_acc1']=test_acc
    val_max = {key:np.round(test_acc[key][test_acc[key].argmax()],3) for key in test_acc}
    
    glb_maxs = {j:max([np.round(metrics[j][i][metrics[j][i].argmax()],3) 
                   for i in metrics[j]]) for j in metrics
    }
    glb_mins = {j:min([np.round(metrics[j][i][metrics[j][i].argmin()],3) 
                   for i in metrics[j]]) for j in metrics
    }
    return metrics, glb_mins, glb_maxs, sorted_keys


def plot(metrics, fid_to_compare=None, title=None):
    
    metrics, mins, maxs, sorted_keys = metrics(fid=fid_to_compare)
    #plt.rcParams['figure.figsize'] = [8,5]
    plt.rcParams['figure.dpi'] = 200
    plt.tight_layout()
    dim = 3
    fig, ax = plt.subplots(1,3,figsize=(22,7))
    fig.patch.set_facecolor('xkcd:white')
    coords = [[i,0] for i in range(dim)]
    metric_types = [[i,j]for i in ['train_loss','validation'] for j in ['acc', 'loss']]

    legend = sorted_keys
    for idx,key in enumerate(metrics):
        metric = metrics[key]
        legend = [name.replace('box','Salient_Patch').strip('_log')
                  for name in metric.keys()]
        
   
        coord = coords[idx]
        low_, high_ = 0,0
        for i in metric:
            low, high = (metric[i][np.argmin(metric[i])], 
                         metric[i][np.argmax(metric[i])])
         
            print(low,high)
            ax_title = key.replace('test','validation').replace('1','.').replace('_',' ')
            ax[coord[0]].plot(range(1,len(metric[i])+1),metric[i])
            ax[coord[0]].grid(which='both')
            ax[coord[0]].grid(which='minor',alpha=0.3)
            ax[coord[0]].grid(which='major',alpha=0.9)
            ax[coord[0]].set_yticks(np.linspace(np.round(mins[key],1)-.05,
                                                np.round(maxs[key],1)+.05,10))
            ax[coord[0]].set_xticks(np.linspace(1,10,10))
            ax[coord[0]].set_ylabel(ax_title, fontsize=18)
            ax[coord[0]].set_xlabel('Epoch', fontsize=18)
            ax[coord[0]].legend(legend, fontsize=10)
           
            ax[coord[0]].set_title(ax_title, fontsize=20)
        fig.suptitle(title, fontsize=20)
    fig.savefig(str(title)+'.png')

    plt.show()

class From_Drive():
    def __init__(self,**kwargs):
        self.urls = [ ]
        fids = [fid.path for fid  in os.scandir() if 'txt' in fid.name]
        for fid in fids:
            with open(fid,'r') as txt_fid:
                self.urls = txt_fid.readlines()
                    
        self.file_keys = [url.split('/')[5] for url in self.urls]
    def google_getter(self):
        
            for f_key in self.file_keys:
                gdd.download_file_from_google_drive(file_id=f_key,
                                                        dest_path='/metrics',
                                                    unzip=False)
            files = [file.path for file in os.scandir()]
            for file in files:
                print(f'\n The files are : {file}')

class Results:
    def __init__(self):
        self.metrics = {fid_.name:self.flatten(fid_)[0] for fid_ in os.scandir('metrics') 
           if  'json' in fid_.name and 'all' not in fid_.name}
        self.metrics_max =  {fid_.name:self.flatten(fid_)[1] for fid_ in os.scandir('metrics') if  'json' in fid_.name}
        
    def flatten(self,fid_):
        with open(fid_, 'r') as fid:
            results_dict = json.load(fid)
        keys = list(results_dict.keys())
        results_arrays_dict = { }
        results_max = { }
        for phase in ['validation', 'training']:
            try:
                results = np.stack([[results_dict[key][phase+' loss'], 
                results_dict[key][phase+' acc'], 
                results_dict[key][phase+' ballance_acc']] for key in keys if phase in key], axis=0)
                results_arrays_dict[phase]= {'loss':results[...,0], 
                                             'acc':results[...,1], 
                                             'ballanced_acc':results[...,2]}
                results_max[phase] = {i:
                    np.round(results_arrays_dict[phase][i][np.argmax(results_arrays_dict[phase][i])],3)
                    for i in results_arrays_dict[phase] if 'acc' in i}

            except:
                print(f'{fid_} not parsed to np array')


        return results_arrays_dict, results_max

def net_plot(all_metrics, epo):
    #plt.rcParams['figure.figsize'] = [21,7]
    plt.rcParams['figure.dpi'] = 200
    plt.tight_layout()
    phases, metrics_type = ['validation', 'training'], ['acc','loss', 'ballanced_acc']
    combinations = [[i,j] for i in phases for j in metrics_type]

    dim = np.floor_divide(len(all_metrics),1)
    fig, ax = plt.subplots(1,dim,figsize=(18,6))
    fig.patch.set_facecolor('xkcd:white')
    print(ax)
    if hasattr(ax, '__iter__'):
        print('ax=',ax)
        if len(ax.shape)==1:
            for idx,metrics_key in enumerate(all_metrics):
                print(idx)
                history = all_metrics[metrics_key]
                
                for i in combinations:
                    #print(np.arange(len(history[i[0]][i[1]][:10]))+1,history[i[0]][i[1]][:10])
                    ax[idx].plot(np.arange(len(history[i[0]][i[1]][:epo]))
                                 +1,history[i[0]][i[1]][:epo])
                    ax[idx].grid(which='both')
                    ax[idx].grid(which='minor',alpha=0.9)
                    ax[idx].grid(which='major',alpha=0.9)
                    ax[idx].set_yticks(np.linspace(0,1,10))
                    #print(np.arange(len(history[i[0]][i[1]][:10]))+1)
                    ax[idx].set_xticks(np.arange(len(history[i[0]][i[1]][:epo]))+1)
                    ax[idx].set_ylabel('Accuracy/Loss', fontsize=12)
                    ax[idx].set_xlabel('Epoch', fontsize=12)
                    ax[idx].legend(combinations)
                    title = ' '.join(
                        [fnm.capitalize() for fnm in metrics_key.split('_')[:2]])
                    ax[idx].set_title(title, fontsize=20)
        fig.suptitle('Convolutional Transformer', fontsize=20)
        plt.savefig('CVT')

class Evaluate:
    def __init__(self,json_fid):
        self.eval_metrics = { }
        self.fid = json_fid
        self.get_dict()
        
    def get_dict(self):
        eval_metrics = { }
        with open(self.fid, 'r') as fid:
            self.results_dict = json.load(fid)
    
    def get_one(self):
        for mod_key in self.results_dict:
            #each model
            model = results_dict[mod_key]
            im_clss = np.array([[model[key]['pred_class'],model[key]['g_t_class']] 
                      for key in model if 'test_acc' not in key])
            yield im_clss[...,0],im_clss[...,1],mod_key
            
    
    def get_ballanced(self,y_pred,y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        return (tpr+fpr)/2
        
    
    def eval_all(self):
        for y_pred,y_true,mod_key in self.get_one():
            acc = accuracy_score(y_pred,y_true, normalize=True)
            b_acc = self.get_ballanced(y_pred,y_true)
            f_one = f1_score(y_pred,y_true, average='macro')
            self.eval_metrics[mod_key]= {'Accuracy':acc.round(4)*100,
                                    'Ballanced Acc.':b_acc.round(4)*100,
                                    'F1':f_one.round(4)*100}
        
    def to_df(self):
        df = pd.DataFrame(self.eval_metrics).T
        if 'Accuracy' in df:
            df = df.sort_values(by='Accuracy')
        latex_tab = df.to_latex()
        print(latex_tab)
        return df
    
    def get_confusion(self):
        self.eval_metrics = { }
        get = self.get_one()
        for y_pred,y_true,mod_key in get:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            self.eval_metrics[mod_key]={
                'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp
            }
            
class Uniques:
    def __init__(self,fid):
        self.labels = ['tn','fn','fp','tp']
        self.fid = fid
    
    def unique(self):
        model_sets = self.get_ims()
        print(model_sets['resnet_152'].keys())
        metrics_grouped = self.get_unique(model_sets)
        return metrics_grouped
            
        
    def get_ims(self):
        with open(self.fid, 'r') as fid:
            results_dict = json.load(fid)
        model_sets = { }
        for mod_key in results_dict:
            print(mod_key)
            model = results_dict[mod_key]

            ims = np.array([[key,model[key]['pred_class'],model[key]['g_t_class']] 
                      for key in model if 'test_acc' not in key])

            sets = self.get_sets(ims)
            print(mod_key)
            print(sets.keys())
            model_sets[mod_key] = sets 
        return model_sets
    
    def get_sets(self,ims):
        bins = [[i,j] for i in [0,1] for j in [0,1]]
        sets = { }
        for sub_set,lab in zip(bins,self.labels):
            idx = np.intersect1d(
            np.where(ims[...,2].astype(int)==sub_set[0]),
            np.where(ims[...,1].astype(int)==sub_set[1]))
            sets[lab]=set(ims[idx][...,0].tolist())
        return sets

    def get_unique(self,sets_dict):
        labels = { }
        for pred_type in self.labels:
            models = { }
            labels[pred_type]=models
            for model_key in sets_dict:
                
                sets = [sets_dict[key][pred_type] 
                        for key in sets_dict if  key!=model_key]
                models[model_key]=list(sets_dict[model_key][pred_type]-set().union(*sets))
        
        return labels  
 