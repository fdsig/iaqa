#### html_content = requests.get(url).text

import requests
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

class html_scraper():
    
    def __init__(self):
       
        self.scraperapi = """'insert api key generate by 
        http://api.scraperapi.com'"""
        wd = '/ava_scraper'
        
        data_dir = os.getcwd()+'/data/'
        print(f'dat dir = :-) {data_dir}')
        failed = 'failed.txt'
        if wd not in os.getcwd():
            wd = os.getcwd()+wd
            os.mkdir(wd)
            os.chdir(wd)
        if data_dir not in list(os.scandir()):
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            with open(data_dir+failed, 'w') as fid:
                fid.write('FAILED PAGE SCRAPES')

        self.fid = 'full_dp_challenge_dot_com_data_dict'
        self.data_dict = { }
        self.img_url_base = 'https://www.dpchallenge.com/image.php?IMAGE_ID='
        self.scrape_type = None



        with open(data_dir+'/failed.txt','r') as fid:
            self.Request_failed = fid.read()
        for fid in [fi_d.name for fi_d in os.scandir()]:
            if self.fid in fid:
                with open(fid,'r') as fid:
                    print('data loaded')
                    self.data_dict = json.load(fid)
        if 'data_dict' not in vars(self):
            self.data_dict = { }
        
        print(f'len data dict = {len(self.data_dict)}')
        self.get_new()
            
        
    def image_url(self,page):
    # Parse the html content
        base = 'https://www.dpchallenge.com'
        url = page
        im_id = page.split('=')[-1]
        payload = {'api_key': self.scraperapi, 'url':url}
        response = requests.get('http://api.scraperapi.com', params=payload).text
        soup = BeautifulSoup(response, 'html.parser')
        image_soup = soup.find_all('img')
        image_src = iter({i['src'] for i in image_soup if im_id in i['src']})
        metrics = soup.find_all('td',{'class':"textsm", 'valign':"top"})
        metrics = [
            i.get_text() for i in metrics if 'Avg' in i.get_text()
                    ][0].split('\n')
        im_title = soup.title.get_text()
        metrics_dict = {i.split(':')[0]:i.split(':')[1] for i in metrics if ''!=i}
        metrics_dict['title'] = soup.title.get_text()
        metrics_dict['author_comments'] = soup.find_all('td', 
                                        {'class':'textsm',
                                         'width':"450",
                                        'valign':"top"
                                        }
                                                       )[0].get_text()
        metrics_dict['forum comments'] = [
            comment.get_text() for comment in soup.find_all(
                'table',{
                    'class':"forum-post"
                }
            )
        ]
        metrics_dict = {im_id:metrics_dict}
        print(metrics_dict)
        return {'image_url':next(image_src), 
                'meta_data':metrics_dict, 
                'id':im_id}

    def get_challenges(self):
        url = """https://www.dpchallenge.com/challenge_history.php?
        order_by=0d&open=1&member=1&speed=1&invitational=1&show_all=1"""
        payload = {'api_key': self.scraperapi, 'url':url}
        response = requests.get('http://api.scraperapi.com', params=payload).text
        soup = BeautifulSoup(response, 'html.parser')
        links = soup.find_all('a')
        base = 'https://www.dpchallenge.com'
        return [base+challenge['href']+'&amp;show_full=1' for challenge in links
                  if 'CHALLENGE_ID' in challenge['href']][::-1]

    def competition_image_links(self):
        base = 'https://www.dpchallenge.com'
        url = self.challenge
        payload = {'api_key': self.scraperapi, 'url':url}
        response = requests.get('http://api.scraperapi.com', params=payload).text
        soup = BeautifulSoup(response, 'html.parser')
        links = soup.find_all('a')
        return(soup,list({base+link['href'] for link in links if 'IMAGE' in link['href']}))

    def get_image(self,image_dict): 
        pass #FINISH THIS CODE
        base = 'https:'
        url = base+kwargs['image_url']
        payload = {'api_key': self.scraperapi, 'url':url}
        response = requests.get('http://api.scraperapi.com', stream=True,params=payload)
        fid_name = 'ava_images_new/'+kwargs['id']+'.jpg'
        with open(fid_name, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)

    def to_json(self,meta_data, open_as):
        with open(self.fid+'.json', open_as) as json_fid:
            data_dict = json.load(json_fid)
            try:
                data_dict.update(kwargs['meta_data'])
                data_dict = json.dumps(data_dict, indent = 4)
                json_fid.seek(0)
                json.dump(data_dict,json_fid)
            except:
                print('json not fully loded')
                data_dict = json.loads(data_dict)
                data_dict.update(kwargs['meta_data'])
                data_dict = json.dumps(data_dict, indent = 4)
                json_fid.seek(0)
                json.dump(data_dict,json_fid)
            finally:
                print(url,' : not scraped')

    def dp_write(self):    
        with open(self.fid +'.json', 'w') as fid:
            data = json.dump(self.data_dict,fid, sort_keys=True, indent=4)

    def get_new(self):
        self.current = np.array([i for i in self.data_dict if i[0].isdigit()])
        print(f'challenges attepted {len(self.current)}')
        re_scrape = np.array([
            chall_idx for chall_idx in self.data_dict  
            if 'fail message' in self.data_dict[chall_idx] 
            or  'error' in self.data_dict[chall_idx]])
        self.current = np.setdiff1d(self.current, re_scrape)
        print(f'succesfully scraped {len(self.current)} : to rescrape {len(re_scrape)}')

        self.all_challenges = np.array(self.get_challenges())
        self.challs_dict = {[split_url.split('&amp;') for split_url in challange_url.split('=')][1][0]
          :challange_url for challange_url in self.all_challenges}
        self.data_dict['challenge_urls']= self.challs_dict
        self.all_id_challenges = np.array(list(self.challs_dict.keys()))
        print('all_challenges=',len(self.all_challenges))
        self.new_challenges = np.setdiff1d(self.all_id_challenges, self.current)
        print(f' unscraped challenges = {len(self.new_challenges)}')
        np.random.shuffle(self.new_challenges)


    def parse_html(self):
        links = [
        html_table for html_table in 
        self.soup.find_all('td', align="center", valign="middle", width="160")
        ]
        li = [
            'no_url_available' if  html.img['src']==None else html.img['src'] 
            for html in links
        ]

        image_id = [
            'removed' if 'removed' in id_ else id_.split('_')[-1] 
            for id_ in li
        ]

        tables = [
            'missing' if html_element ==None else html_element.next_sibling.next_sibling
            for html_element in self.soup.find_all(
                'td', align="center", 
                valign="middle", 
                width="160")]

        meta = [
            None if html.span == None else 
            html.find('td', valign="top").next_sibling.next_sibling.get_text() 
            for html in tables
        ]
        
        meta = [
            ['no_values']*4 if mean==None else[
                mean.split('\t')[-7].strip('\n').split(':') +
                mean.split('\t')[-1].strip('\n').split(':')][0] 
             for mean in meta
        ]
        
        

        total_means = [
                None if html_element.b == None
                else list(html_element.b.next_siblings)
                for html_element in tables
        ]
        total_means = [
            0 if len(vote)==3 else 
            None if  'NavigableString' in str(type(vote[3])) 
            else vote[3].get_text()
            for vote in total_means
        ]

        
        title = self.soup.div.next_sibling.next_sibling.b.get_text()
        
        comp_data = {
            comp_item.string:[comp_item.next_sibling.string,
            comp_item.next_sibling.next_sibling.string] 
            for comp_item in [div.find_all('b') 
            for div in self.soup.find_all('div', style="margin: 2px;")][0]
        }
        
        
        div = [
            item for item in self.soup.find_all('div')
        ]
        
        description = [
            description.next_siblings for description in [
               item.b.next_sibling for item in div if item.b !=None
        ] if description!=None][:3]
        
        description = [
            sub_item.string.strip('\n').strip('\t').strip('\n') 
            for item in description 
            for sub_item in item if sub_item.string!=None
        ]

        comp_data['title']=title; comp_data['description']=description
        
        titles_c = [
           ['missing','missing'] if a_html.a == None else 
           [a_html.a['href'],a_html.a.get_text()] 
           for a_html in tables
        ]
        votes = [
            None if table.span ==None else 
            list(table.span.next_siblings)[-3].get_text() 
            for table in tables
        ]
        views_total = [
            None if table.span == None else 
            list(table.span.next_siblings)[1].get_text()
            for table in tables
        ]
        
        views = [
            None if table.span == None else 
            list(table.span.next_siblings)[2].get_text()
            for table in tables
        ]
        
   
        rank = [ None if i.b ==None else 
           i.b.get_text() for i in self.soup.find_all(
            'td', valign="top", width="200")
        ]
     

        self.meta =  { 'titles':titles_c,
            'meta_data':meta,'total_means':total_means, 
            'image_ids':image_id,'url':li, 'competion_details':comp_data,
            'rank':rank, 'n_votes':votes,'n_views':views_total,
            'views_during_voting':views
                     }


    def get_data_dict(self):
        self.new_id = self.new_challenges
        self.all_id = self.all_challenges
        self.old_id = np.setdiff1d(self.all_id_challenges, self.new_challenges)
        if self.scrape_type==None and 'data_dict' not in vars(self):
            with open(self.fid + '.json', 'r') as fid:
                self.data_dict = json.load(fid)
                self.challs_touched = list(self.data_dict.keys())
        self.data_dict['scraped']= { }
        rng = np.random.default_rng()
        self.new_challenges = rng.permutation(self.new_challenges)
        for chall in tqdm(self.new_challenges, position=0, leave=True):
            self.chall_id = chall
            self.challenge = self.challs_dict[chall]
            if chall in self.new_challenges:    
                self.data_dict['scraped'][self.chall_id] = self.chall_id in self.new_challenges
                self.soup, self.image_pages = self.competition_image_links()
                self.text = self.soup.get_text()
                if self.soup.string == self.Request_failed or 'Request failed' in self.text:
                    self.data_dict[self.chall_id]=self.Request_failed
                    self.all_to_dict(success=False)
                       
                else:
                    self.parse_html()
                    self.all_to_dict(success=True)
                   
                test = len(self.new_challenges)
                idx_del = np.where(self.new_challenges == self.chall_id)
                self.new_challenges = np.delete(self.new_challenges,idx_del)
                self.new_challenges = rng.permutation(self.new_challenges)
            else:
                continue
                
                    
            while len(self.new_challenges)==test:
                try:
                    with open(self.fid + '.json', 'r') as fid:
                        self.data_dict = json.load(fid)
                        self.challs_touched = np.array(list(data_dict.keys()))
                        self.new_challenges = np.setdiff1d(self.challs_touched, self.all_challenges)

                except:
                    print('not read')

            
    
    def all_to_dict(self,**kwargs):
        if kwargs['success']:
            chall_id = self.chall_id
            self.data_dict[chall_id]={
                self.meta['image_ids'][idx]+'_'+str(idx):
                {'urls':{'thumbnail':
                         'https:'+ self.meta['url'][idx],
                         'full_resolution': 'https:'+
                         self.meta['url'][idx].replace('/120/','/1200/')},  
                 'meta':{ 'means':self.meta['meta_data'][idx],
                          'Overall_Mean':self.meta['total_means'][idx],
                          'votes':self.meta['n_votes'][idx],
                          'rank':self.meta['rank'][idx],
                         'views_all':str(self.meta['n_views'][idx]),
                         'view_during_voting':str(self.meta['views_during_voting'][idx])
                        },
                'image'+self.meta['image_ids'][idx]+'_page':
                 self.img_url_base+self.meta['image_ids'][idx].strip('.jpg'),
                 'image_title':self.meta['titles'][idx]
                } for idx in range(len(self.meta['image_ids']))}
            self.data_dict[chall_id]['status'] = {
                'scrape_items_match':
                len(self.meta['url'])==len(self.meta['n_votes'])==len(self.meta['meta_data']),
                'subset_new':chall_id in self.new_id, 
                'subset_old':chall_id in self.old_id,
                'subset_all':chall_id in self.all_id
            }
            self.data_dict[chall_id]['all_urls'] = {
                url.split('=')[-1]:url for url in self.image_pages
            }
            self.data_dict['scraped'][chall_id] = chall_id in self.new_challenges
            self.data_dict[chall_id]['comp_data']=self.meta['competion_details']
            self.fid='dp_challenge_dot_com_data_dict'
            self.dp_write()
        else:
            self.data_dict[self.chall_id]={'fail message':self.text}
            self.data_dict[self.chall_id]['status'] = {
                'subset_new':self.chall_id in self.new_id, 
                'subset_old':self.chall_id in self.old_id,
                'subset_all':self.chall_id in self.all_id
            }
            self.dp_write()


