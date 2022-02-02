import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import cv2
import os
import json
import data_dict as dd
class Plot:
    def __init__(self):
        self.flip=False; 
        self.scalar=0.07; 
        self.text_scalar=1.2; 
        self.sub_text_scalar=1.2
        self.dest = 'overall_dataset/'
        self.n_examples = 4
        self.rotate = None
        
    def scale(self, **kwargs):
        self.scalar=kwargs['scalar']
        self.text_scalar=kwargs['text_scalar']
        self.subtext_scalar=kwargs['subtext_scalar']
        
    def image_resize(self, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        image = self.img
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def image_grid_plot(self, x=None,y=None, dataset=None, title=None, group_by_data=None,
                       stack=None):
        if x is None:
            x = 1
        if y is None:
            y = 1 
        print(x,y)
        im_read = lambda img:cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        image_fids_stack = [fid_1 for fid in os.scandir(dataset) 
            for fid_1 in os.scandir(fid)]

        if group_by_data:
            image_fids_stack = [list(os.scandir(fid_ ))[:self.n_examples] 
                                for fid in image_fids_stack 
                            for fid_ in os.scandir(fid)]
            
        else:
            image_fids_stack = [[k for j in os.scandir(fid) 
                                 for k in os.scandir(j)][:self.n_examples]
                                for fid in image_fids_stack]
        
     
 
        image_fids = [j for i in image_fids_stack for j in i]
        fid_details = [fid.path.split('/') for fid in image_fids]
        titles = [fid.name for fid in image_fids]
        xlab = [x_name[-2] for x_name in fid_details]
        data_types = set([sub_t[2]+'_'+sub_t[3] for sub_t in fid_details])
        classes = set(xlab)

        img_arr = [im_read(i.path) for i in image_fids]
    
    
        if self.flip:
            img_arr = [cv2.flip(im,0) for im in img_arr]
        mean_h = np.ceil(np.mean([array.shape[0] for array in img_arr])).astype(int)
        mean_w = np.ceil(np.mean([array.shape[1] for array in img_arr])).astype(int)
        for idx,img in enumerate(img_arr):
            self.img = img
            img_arr[idx] = self.image_resize(width=mean_w)
            
        if stack:
            slice_by = len(image_fids_stack[0])//2
            image_fids_stack = [image_fids_stack[0][:slice_by],
                                image_fids_stack[0][slice_by:]]
        
        dim_y = len(image_fids_stack)
        if np.sqrt(dim_y)% 2 == 0:
            dim_y = dim_x = int(dim_y)

        else:
            
            dim_x = len(image_fids_stack[0])
        
        if self.rotate:
    
            temp = dim_y
            dim_y = dim_x
            dim_x = temp
            
        if len(img_arr)%2!=0:
                blank = np.full((mean_h,mean_w,3),255)
                img_arr+=[blank]
                titles+='*_____*'
                xlab+='_____'
        

        fig_wdth = int((mean_w*len(image_fids_stack[0])*x//10)*self.scalar)
        fig_hght =  int((mean_h*y//10)*self.scalar)
        print(fig_wdth,fig_hght)
        

        fig, ax = plt.subplots(dim_y,dim_x, figsize=(fig_wdth,fig_hght))
        fig.patch.set_facecolor('xkcd:white')
        dim_idx = 0
        if hasattr(ax, '__iter__'):
            if len(ax.shape)==1:
                for idx,sub_x in enumerate(ax):
                    for idx_, sub_y in enumerate(img_arr):
                        img = img_arr[idx]
                        sub_x.imshow(img)
                        sub_x.set_xticks([])
                        sub_x.set_yticks([])
                        sub_x.set_title(titles[idx], 
                                        size=10*self.subtext_scalar,
                                        pad=10*self.text_scalar)
   
                        sub_x.set_xlabel(f'{xlab[idx].capitalize()}',color='blue', 
                                         size=12*self.subtext_scalar)
                        sub_x.spines['top'].set_visible(False)
                        sub_x.spines['right'].set_visible(False)
                        sub_x.spines['bottom'].set_visible(False)
                        sub_x.spines['left'].set_visible(False)
                       
                        dim_idx+=1

            else:
                for idx,sub_x in enumerate(ax):
                    for idx_, sub_y in enumerate(sub_x):
                        img = img_arr[dim_idx]
                        sub_x[idx_].imshow(img)
                        sub_x[idx_].set_xticks([])
                        sub_x[idx_].set_yticks([])
                        sub_x[idx_].spines['top'].set_visible(False)
                        sub_x[idx_].spines['right'].set_visible(False)
                        sub_x[idx_].spines['bottom'].set_visible(False)
                        sub_x[idx_].spines['left'].set_visible(False)
                        sub_x[idx_].set_title(titles[dim_idx], 
                                              size=10*self.subtext_scalar
                                              ,pad=5*self.text_scalar)
                        sub_x[idx_].set_xlabel(xlab[dim_idx].capitalize(),
                                               color='blue', 
                                               size=15*self.subtext_scalar)
                        dim_idx+=1
        else:
            img = img_arr[0]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title(titles[dim_idx].capitalize(), 
                         size=15*self.subtext_scalar,
                         pad=15)
        
        
        fig.supxlabel(f'Data Type :{data_types}\
        \n Ground Truth : {classes}', 
                  fontsize=15*self.text_scalar)
        fig.suptitle(title, size=20*self.text_scalar)
        plt.savefig(self.dest+title+'.png',dpi=70)
        
def order_name():
    start_n = 0
    for idx,did in enumerate(os.scandir('data_sets/')):
        fids = enumerate([
       l for i in os.scandir(did) 
         for j in os.scandir(i) 
         for k in os.scandir(j)
         for l in os.scandir(k)])

        old_files = [l.name
            for i in os.scandir(did) 
            for j in os.scandir(i) 
            for k in os.scandir(j)
            for l in os.scandir(k)]


        for fid in fids:
            start_n+=1
            old_fid , path_new = (fid[1].path, '/'.join(fid[1].path.split('/')[:-1])
                       +'/')
            new_name = (fid[1].path.split('/')[1]
                       +'_'
                       + str(fid[0])
                       +"_"
                       +str(len(old_files))
                       +'_'
                       +str(start_n)
                       +'.'
                       +fid[1].name.split('.')[-1])
            if new_name in old_files:
                check_index = 0
                while new_name in old_files:
                    check_index+=1
                    name = new_name.split('.')[0]
                    ext = new_name.split('.')[1]
                    name = name+'_'+str(check_index)
                    new_name = '.'.join([name,ext])
            new_fid = path_new+new_name
            os.rename(old_fid,new_fid)
class Metrics_plot:
    def __init__(self):
        self.data_dict = dd.Read().data
    def plot(self):
        data_dict = self.data_dict
        from adjustText import adjust_text
        plt.style.use('bmh')
        size_year = np.array(
            [[int(data_dict[i]['year']),int(data_dict[i]['size'])]
             for i in data_dict])
        data_source = [data_dict[i]['annote'] for i in data_dict]
        x = size_year[...,0].astype(int)
        y = size_year[...,1].astype(int)
        y_log = (np.log(y))
        fig, ax = plt.subplots(figsize=(12,8))
        fig.patch.set_facecolor('xkcd:white')
        std,mean = np.std(y),np.mean(y)

        prcc, pval = stats.spearmanr(x,y)
        print(prcc)
        for i, txt in enumerate(size_year):
            ax.scatter(x[i],y_log[i], alpha=0.3, s=1.9**y_log[i])
            #ax.annotate(data_source[i]+' ('+str(y[i])+')',(x[i],y_log[i]), )

            ax.set_ylabel('log data set size', size=18)
            ax.set_ylim(6,16)
            ax.set_xlabel('Year Dataset Compiled', size=18)
            ax.legend(data_source, markerscale=0.2,loc='upper left')
        fig.supxlabel(f'PRCC {abs(prcc):.2f}  p value {pval:.2f}', size=20)
        fig.suptitle('IAQA Datasets Scatter', size=25)
        texts = [ax.text(x[i], y_log[i], data_source[i]) for i in range(len(x))]
        #adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))

        adjust_text(texts, force_points=0.5, force_text=0.3,
                    expand_points=(2, -2), expand_text=(-2, -2),
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
        print(mean)
        plt.savefig('iaqa_ds_size',dpi=200)
