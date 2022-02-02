from google_drive_downloader import GoogleDriveDownloader as gdd

import os
import zipfile
import shutil
from tqdm import tqdm

class Get_Ava:

    # a fully autmated downlaod pipeline in two flavours:
    # Vanilla: manually mounting own drive using read type = 'own_drive'
    # ava batches (all links in ava_files_urls.txt) this will take 
    # 32 GB of google drive

    # Salted Caramel = simply runnign code with read_type = 'public'
    # this does not require you to have any of the ava Benchmark files
    # will require 96GB of google colab apace 64Gb + 32Gb (while unzipping)
    # after zips deleted 64 GB
    # it would be possible to eaisilty adapt code to dowload unzip 
    # delete individula batches thus shrinking 32gb overhead
    # to 800 mb per batch total space requried 65 GB

    def __init__(self,**kwargs):
        ''' dowloads models and ava images form gooogle dive (no mounting)'''
        drive.flush_and_unmount()
        os.system('git clone https://github.com/Openning07/MPADA.git')
        self.google_getter
        self.download_ava_files 
        

    def replace_ava(self):
        
        
        self.mpada_git = [os.path.join(root,name) for root,dirs,files in os.walk('/content/MPADA/', topdown=False) for name in files]
        self.origional_py_files = [path for path in self.mpada_git if path[-2:]=='py']
        self.origional_py_names = [path.split('/')[-1] for path in self.mpada_git if path[-2:]=='py']
        check = input('replace .py files y/n:  ')
        while not any((check=='y', check=='n')):
            check = input('not y or n try y or n:')
            print(check)

        y, n = check == "y", check == "n"
        print(f' you entered n = No:  {n} , you entered y = Yes {y}' )
        if check=='y':
            for py in self.origional_py_files:
                os.remove(py)
                print(f' orgional py files:\n {py}\n removed')
            new_py = [py_file.path for py_file in os.scandir('/content/mpada_temp/')]
            import shutil
            print(new_py)
            for old,new in zip(self.origional_py_files , new_py):
                print(f'{new} \nhas been moved to \n{old}')
                shutil.copy(new,old)
        else:
            print('no files moved')


    def cleanup(self):  
        os.chdir('/content/')
        acceptable_dir = ['MPADA', 'mpada_temp', 'Images', 'models','.config']
        to_clean  = [root_dir.path for root_dir in os.scandir('/content/') if root_dir.name not in acceptable_dir]
        clean_command = 'rm -rf'
        if os.getcwd()=='/content/':
            for dir_ in to_clean:
                os.system(clean_command+' '+dir_)

    def parse_urls(self):
        self.args = {'zip':False}
        # get a list of urls to dowload whole detaset - the url below is itself a .txt of ava batch zips
        # which are parsed in code cell five
        # this is use by 'get data()' to pull all 255000 images
        ava_batches_urls = ['https://drive.google.com/file/d/1YtU0m8cf2qgYcSpcPqlSz2GxO7wowu6W/view?usp=sharing']
        self.args['urls']=ava_batches_urls
        self.args['file_names']=['ava_files_urls.txt']
        self.args['dest_path']= './batch_meta/'
        






    def google_getter(self):

        file_keys = [url.split('/')[3].split('=')[1] for url in self.args['urls'] 
                    if 'usp=sharing' not in url]
        
        for link in self.args['urls']:
            print(f'{link}\n')
        file_keys += [url.split('/')[-2] for url in self.args['urls'] if 'usp' in url]

        for f_key,fid in zip(file_keys,self.args['file_names']):
            gdd.download_file_from_google_drive(file_id=f_key,
                                                    dest_path=self.args['dest_path']+'/'+fid,
                                                unzip=self.args['zip'])
        files = [file.path for file in os.scandir(self.args['dest_path'])]
        for file in files:
            print(f'\n The files are : {file}')
    
    def ava_txt(self):
        """this fucntin reads from frida de sigleys google drive usin share keys"""
        with open('/content/batch_meta/ava_files_urls.txt', 'r') as fb:
            txt = fb.readlines()
        self.files_crypt = [i.split('/')[-2] for i in txt[0].split(',')]
        if len(self.files_crypt)==44:
            print(f'File IDs have sucessfully been obtaniend and now in content/batch_meta/')
        print(self.files_crypt)
       
            
    
    def download_ava_files(self,**kwargs):
        
        '''this funciont getts all ava files in 44 patches
        a prime factor of n imges in whole data set
        all batchs have equal n images 255,000/44'''
        ''' you can give your own google drive 
        but his is not needed as it will parse and
        dowload form public google drvie file ID
        for FRIDA de Sigley aka Pale Flower's googel drive'''
        """ this will take some time(10 mins)- you canve view progress by 
        clicking on file incon tab and going to IMAGES"""
        '''The script unzips and them moves all image files into 
        All directory and deletes the zip files and the unzipped batches
        --- a fully autmated pipeline with no need for google drive mounting'''
        print(kwargs)
        if kwargs['clear_current']:
            os.system('rm -rf Images')
        else:
            print('no files cleared if re running this ') 
        try:
            os.mkdir('Images')
        except:
            print('exists')
        finally:
            if not os.path.exists('Images/images'):
                os.mkdir('Images/images')  
            else:
                print('Images/images exists also')

        if kwargs['own_drive']:
            drive.mount('/content/drive', force_remount=True)
            self.paths = [i.path for 
                          i in 
                          os.scandir('/content/drive/MyDrive/0.AVA/AVA_dataset/batches_zip/') 
            if '(Unzipped Files)' not in i.path and i.path[-3:]=='zip']
            
            if kwargs['full']==False:
                paths = self.paths[-4:]
                print(paths)
            else:
                paths = self.paths

            for zip_f in tqdm(paths, colour=('#FF69B4')):
                zip_ref = zipfile.ZipFile(zip_f, 'r')
                zip_ref.extractall("./Images/")
                zip_ref.close()

            paths = [j.path for i in os.scandir('Images') 
            if 'Batch' in i.name for j in os.scandir(i)]
            existing_fids = [ ]

            for i in tqdm(paths):
                try:
                    shutil.move(i,'Images/images')
                except:
                    existing_fids.append(i)
            print(f'{len(existing_fids)} already exist \n all files = {len(paths)}')


            for path in tqdm(os.scandir('Images')):
                if 'Batch' in path.name:
                    os.rmdir(path.path) 
            print(len(list(os.scandir('Images/images/'))), 
                  list(os.scandir('Images')))
        
        else:
            if kwargs['full']==False:
                self.files_crypt = self.files_crypt[-4:]
                print(self.files_crypt)

            current = [i.name for i in os.scandir('Images/')]
            if kwargs['download']:
                for id_ in tqdm(self.files_crypt, colour=('#FF69B4')):
                    if id_ not in (current):
                        gdd.download_file_from_google_drive(file_id=id_,
                                                dest_path='./Images/'+id_,
                                                unzip=True)
            for po_file in os.scandir('./Images/'):
                if os.path.isfile(po_file.path):
                    os.remove(po_file.path)

            paths = [j.path for i in os.scandir('Images') for j in os.scandir(i)]
            ava_all = len(list(os.scandir('./Images/images')))
            if ava_all<255508 or ava_all==0:
                for po_file in paths:
                    shutil.move(po_file,'Images/images')
            else:
                print(f'only {ava_all} files have dowloaded properly')

            for path in os.scandir('./Images/'):
                if 'Batch' in path.name:
                    #uses linux bash rm -rf to remove sub batches
                    # they should be empty- but if you re run cell
                    # by mistake and part of the above runs
                    # you will have duplicate in batch directorys
                    # therfor recursive delete necessary
                    # these images will aready be in Images/All dir                
                    os.system('rm -rf '+path.path) 
            print(len(list(os.scandir('Images/images/'))), list(os.scandir('Images')))
    ######## (Change from False to True and  read to now_own to use)