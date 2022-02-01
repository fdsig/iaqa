
import classifier
import argparse


#hugging face api autonlp argugments (passed back to lower level in stack) 

parser = argparse.ArgumentParser(description='Training Calssifier ')


parser.add_argument('--input csv', default='gender-classifier-DFE-791531.csv', 
                    type=str, help='relative loacation of input csv for training')
parser.add_argument('--project', default='gender_class', type=str, help='poject name')


parser.add_argument('--split', default='train', type=str, help='dataset split')

parser.add_argument('--col_mapping',default=None, type=str, help='text:text, label:target')

parser.add_argument('--files',default='gender_text_train.csv', type=str, 
                    help='formated csv only 2 colls one for text one for target')

parser.add_argument('--api_key', default=None, type=str, help='api key from hugging_face account')
parser.add_argument('--resize', type=int, help='Resizes images by percentage as scalar')


parser.add_argument('--name', type=str, help='project name hugging face')

parser.add_argument('--language', type=str, help='lang in eg [en,sp,fr]')

parser.add_argument('--task', type=str, default='binary_classification',
                    help='Resizes images by percentage as scalar')
parser.add_argument('--max_models', type=int, default=2, help= 'nuber of trainable models')
parser.add_argument('--create_project', action='store_true',  help='create_new hf project')

#meta args-- directing sub process
parser.add_argument('--hugging_face', action='store_true', help='uses hugging face api to train model')
parser.add_argument('--send', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--login', action='store_true',  help='if entered will try to sen .csv')
parser.add_argument('--make', action='store_true',  help='create_new hf project')
parser.add_argument('--train', action='store_true',  help='create_new hf project')

args = parser.parse_args()


if __name__=='__main__':
    print('here')
    
    if args.hugging_face:
        print(args.api_key)
        classifier.train_hf_api(args)
    
        print('Model trianing using autonlp (hugging face api)')
       

    
   
    
    
    
    
