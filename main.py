import argparse
from train_utils import engine, image_getter, download_ava

#hugging face api autonlp argugments (passed back to lower level in stack) 

parser = argparse.ArgumentParser(description='IAQA research software using CUHK-PQ, AVA and IAD')


#meta args-- directing sub process
parser.add_argument('--download_ava', action='store_true', help='download ava batches')
parser.add_argument('--send', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--login', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--make', action='store_true',  help='create_new hf project')
parser.add_argument('--train', action='store_true',  help='create_new hf project')

args = parser.parse_args()


if __name__=='__main__':
    print('here')
    if args.download_ava:
        download_ava.import_ava()
    
   
    
    
    
    
