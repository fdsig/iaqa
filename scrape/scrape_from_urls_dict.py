import requests
from PIL import Image
import pandas as pd
import random
from bs4 import BeautifulSoup
import os
import urllib.request
import requests
import re
import time
from tqdm import tqdm
import json
import numpy as np

class Image_scrape:
    def __init__(self):  
        self.did = 'dp_challenge_new_images'
        self.json_path = None
        self.not_scraped_dict = { }
        self.url_missformated = { }
        self.base = 'https://images.dpchallenge.com/images_challenge/'        
    def get_meta(self):

        with open(self.json_path, 'r') as fid:
            self.meta_data = json.load(fid)

    def url_from_meta(self,data_dict=None):
        keys_to_exclude = {'imageremoved_page' ,'imagechallenge/hidden.png_page',
                           'subset_all','Average Score: '}
        challenges = [i for i in data_dict.keys() if i.isnumeric()]
        urls = {} 
        print(len(challenges))
        for challenge in challenges:
            im_keys = [data_dict[challenge][i] for i in data_dict[challenge]]
            im_keys = [i for i in im_keys if type(i)==dict] 
            image_urls = [i['link'] for i in im_keys if not set(set(i.keys()) & keys_to_exclude) ]
            for url in image_urls:
                key = url.split('/Copyrighted_Image_Reuse_Prohibited_')[1] 
                urls[key]='https:'+url.replace('/120/','/1200/')
        return urls

    def get_uniqe(self):
        all_image_urls = self.url_from_meta(data_dict=self.meta_data)
        scraped_ids = {fid_id.name for fid_id in os.scandir(self.did)}
        image_ids = set(all_image_urls.keys())
        print(len(image_ids))
        df = pd.read_csv('AVA_data_official_test.csv')
        current_ava = set(df['image_name'].values)
        unique = current_ava ^ image_ids ^ scraped_ids 
        not_unique = set.union( image_ids ^ unique )
        print(len(image_ids),  len(unique), len(not_unique))
        if len(unique)+len(not_unique):
            print(len(unique)+len(not_unique))
            self.to_scrape = {i:all_image_urls[i] for i in unique if i in all_image_urls}
            if not any([i in current_ava for i in self.to_scrape]):
                print(f'''there are not scraped image in to scrap dict, has :{len(self.to_scrape)}
                unique images, total of {len(self.to_scrape)+len(not_unique)}''')
    def scrape(self):
     
        keys = np.array([i for i in self.to_scrape])
        np.random.shuffle(keys)
        constant = 100
        batches = [keys[i-constant:i] for i in range(constant,len(keys),constant)]
        len(batches)

        if not os.path.exists(self.did):
            os.mkdir(self.did)
        for batch in tqdm(batches,colour=('#FF69B4'),position=0,leave=False):
            sleep_for = sum(np.random.random_sample(1))
            time.sleep(sleep_for*60)
            for image in tqdm(batch,colour=('#ff1493'), position=0,leave=False):
                self.fid = self.did+'/'+image
                url = self.to_scrape[image]
                if self.base in url: 
                    try:
                        response = requests.get(url, stream=True).raw
                        im = Image.open(response)
                        im.save(self.fid)
                        time.sleep(sleep_for*2)
                    except:
                        self.not_scraped_dict[image]=url
                        ns_ = 'not_scraped.json'
                        with open(ns_, 'w') as fid: 
                            json.dump(self.not_scraped, fid)
                        print(f'url : {url} not scraped, logged in {ns_} at {image} ')

                else:
                    ns_format = 'missformed.json'
                    print(f'missformatee url = {url}, loged in {ns_format} at {image}')
                    self.url_missformated[image]=url
                    with open('missformeted_urls.json', 'w') as fid:
                        json.dump(self.url_missformated , fid)

scrape = Image_scrape()
scrape.json_path = 'full_dp_challange_data_dict.json'
scrape.get_meta()
scrape.get_uniqe()
scrape.scrape()
