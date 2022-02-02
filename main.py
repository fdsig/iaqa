import argparse
from train_utils import engine, image_getter, download_ava

# hugging face api autonlp argugments (passed back to lower level in stack)

parser = argparse.ArgumentParser(
    description='IAQA research software using CUHK-PQ, AVA and IAD')


# meta args-- directing sub process
parser.add_argument('--download_ava', action='store_true',
                    help='download ava batches')
parser.add_argument('--send', action='store_true',
                    help='if entered will try to sen .csv')
parser.add_argument('--login', action='store_true',
                    help='if entered will try to sen .csv')
parser.add_argument('--make', action='store_true',
                    help='create_new hf project')
parser.add_argument('--train', action='store_true',
                    help='create_new hf project')

args = parser.parse_args()


if __name__ == '__main__':
    print('here')
    if args.download_ava:
        download_ava.import_ava()
    if args.train:
        engine.ava_data_reflect
        engine.get_df()

def get_all():
    '''meta fucntion for calling other fuctions'''
    df = get_df()
    df = meta_process(df=df)
    class_weights, class_counts = class_wts(df['threshold'])
    y_g_dict = get_labels(df)
    make_class_dir(df, y_g_dict)
    y_g_neg = {key: y_g_dict[key]
               for key in y_g_dict if y_g_dict[key]['threshold'] == 0}
    y_g_pos = {key: y_g_dict[key]
               for key in y_g_dict if y_g_dict[key]['threshold'] == 1}
    sets = ['test', 'training', 'validation']
    splits = {
        set_: {
            im_key: y_g_dict[im_key] for im_key in y_g_dict
            if y_g_dict[im_key]['set'] == set_
        } for set_ in sets
    }
    print(
        f"train set n = {len(splits['training'])} \ntest_list n = {len(splits['test'])}\nvalidation_list n = {len(splits['validation'])}")
    return df, y_g_dict, splits, y_g_neg, y_g_pos

