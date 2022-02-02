import os
import zipfile
import shutil
from tqdm import tqdm
zips = '/home/frida/../../media/frida/DATA/AVA/DATA/Batch_Zip/'
full = '/home/frida/../../media/frida/f1/ava_full'
if not os.path.exists(full):
    os.mkdir(full)
zip_fids = [i.path for i in os.scandir(zips) if 'Batch' in i.name]
for zip_f in tqdm(zip_fids):
                zip_ref = zipfile.ZipFile(zip_f, 'r')
                zip_ref.extractall(full)
                zip_ref.close()
                for did in os.scandir(full):
                    if 'Batch_' in did.name and not 'zip' in did.name:
                        for im in os.scandir(did):
                            if not os.path.exists(im.path):
                                shutil.move(im.path,full+im.name)
                            print(len(list(os.scandir(full))))
                            os.rmdir(did.path)
                            print(len(list(os.scandir(full))))
