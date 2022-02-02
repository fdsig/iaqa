import argparse
from train_utils import engine, image_getter

#hugging face api autonlp argugments (passed back to lower level in stack) 

parser = argparse.ArgumentParser(description='Training Calssifier ')


parser.add_argument('--download ava', default='gender-classifier-DFE-791531.csv', 
                    type=str, help='relative loacation of input csv for training')

#meta args-- directing sub process
parser.add_argument('--hugging_face', action='store_true', help='uses hugging face api to train model')
parser.add_argument('--send', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--login', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--make', action='store_true',  help='create_new hf project')
parser.add_argument('--train', action='store_true',  help='create_new hf project')

args = parser.parse_args()


if __name__=='__main__':
    print('here')


def import_ava():

    pull = image_getter.Get_Ava()
    pull.parse_urls()
    pull.google_getter()
    pull.ava_txt()
    pull.download_ava_files(own_drive=False, 
                            download=True, 
                            full=True, 
                            clear_current=False)

import_ava()
    
   
    
    
    
    
