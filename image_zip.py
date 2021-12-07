import os
import zipfile
import cv2
import matplotlib.pyplot as plt
from shutil import move, make_archive
from tqdm import tqdm
class Dp_Image_Arch():
    def __init__(self):
            self.file_mk = 'AVA_Batch/'
    def batch_move(self):
        files = [i.path for i in os.scandir('/home/frida/AVA_ALL_Main_ubuntu/ava_scraper/dp_challenge_new_images/') 
                 if i.name[-3:]=='jpg']
        print(len(files))
        if len(file)!=0:
            batch_factor = len(files)//[i for i in range(1,len(files)+1) 
                    if len(files)%i==0 and i<=100 and i>=10][0]

            chunks = [files[x:x+batch_factor] for x in range(0,len(files), batch_factor)] 
            print(len(chunks))
            file_mk = 'AVA_Batch/'
            try:
                os.mkdir(file_mk)
            except:
                print('dir_exists')
            for i in tqdm(range(len(files)//batch_factor)):
                if not os.path.isdir(file_mk+'/'+'Batch_'+str(i)):
                    os.mkdir(file_mk+'/'+'Batch_'+str(i))
                    print(file_mk+'/'+'Batch_'+str(i))
            dest=[i.path for i in os.scandir(file_mk)]
            for dst,src in zip(os.scandir(file_mk),chunks):
                for src_ in tqdm(src):
                        move(src_,dst)
        else:
            print('no files moved going to batch process')
            self.file_mk = 'AVA_Batch/'
            
    def zip_batch(self):
        zip_dir = '/home/frida/../../media/frida/DATA/AVA/DATA/dp_chall_batch_zip'
        print(f'zipping to {zip_dir}')
        try:
            os.mkdir(zip_dir)
        except:
            print('dir_exists')
        fids = [fid for fid in os.scandir(self.file_mk)]
        for file in tqdm(fids):
            make_archive(zip_dir+'/'+file.name, 'zip', self.file_mk,file.name)
        print(f' ther are {len(os.scandir(zip_dir))} new zips')
        print('zipping finished***********')

print('initialized batch zip')
zipper = Dp_Image_Arch()
zipper.zip_batch()
