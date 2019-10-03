import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import socket
import importlib
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
#TEST_FILES = provider.getDataFiles(\
#    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

TEST_FILES = ['block_rmz.h5']

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, True)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    #log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'pred': pred}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, source_file = provider.loadBlockData(TEST_FILES[fn])
        #current_data = current_data[:,0:NUM_POINT,:]
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        feature_array = np.zeros((0,1024))
        
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                  vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['is_training_pl']: is_training}
                feature = sess.run(ops['pred'],
                                   feed_dict=feed_dict)
                #print(feature.shape)
                feature_array = np.append(feature_array,[np.squeeze(feature)],axis=0)
        
        np.save('feature_0.npy',feature_array)
        np.save('file_list.npy',source_file)
        #create_df
        
        pkl_name = 'feature_'+str(fn)+'.pkl'
        decode_file = [file.decode("utf-8") for file in list(np.squeeze(source_file))]
        df = pd.DataFrame(list(zip(feature_array,decode_file)),columns=['feature','file'])
        df.to_pickle(pkl_name)
        
        
        
if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
