#from __future__ import print_function, division
import os 

import numpy as np
#import tensorflow as tf
#from keras.backend import learning_phase
import h5py
from u_net import Unet_Model
import random


if __name__ == '__main__':
    # parsing
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--dataset-path', type=str, dest='dataset_path', 
                        help='path to `.h5` dataset', required=True)

    parser.add_argument('--input-size', type=int, dest='input_size', 
                        help='size of the input tensor', default=40)

    parser.add_argument('--batch-size', type=int, dest='batch_size', 
                        help='size of a minibatch', default=32)

    parser.add_argument('--epochs', type=int, dest='epochs', 
                        help='number of training epochs', default=30)

    parser.add_argument('--iter-sampler', type=int, dest='iter_sampler', 
                        help='iteration sampler. Either 0 for "uniform" or LAM for "poisson_LAM"\n'
                        'LAM: Lambda parameter in Poisson distribution', 
                        default=0)

    parser.add_argument('--summary-prefix', type=str, dest='summary_prefix', 
                        help='root folder to save the summary', 
                        default='summary')

    parser.add_argument('--save-prefix', type=str, dest='save_prefix', 
                        help='root folder to save the model', 
                        default='trained_models')

    parser.add_argument('--depth', type=int, dest='depth', 
                    help='depth of unet', default='4')

    parser.add_argument('--min-feature', type=int, dest='min_feature', 
                    help='starting number of features', 
                    default='32')

    options = parser.parse_args()
    # training
    model = Unet_Model(u_layers=options.depth,min_feature=options.min_feature)

    print('start')
    
    f = h5py.File(options.dataset_path, 'r')

    targets = np.float32(f['targets'])
    iters=np.float32(f['iters'])
    
    X_np=np.float32(iters)
    
    l,h,w=iters.shape[0],options.input_size,options.input_size
    
    X=np.ndarray(shape=(l,h,w,2), dtype=float)
    y=np.float32(targets)
    
    print('loaded')
    if(options.iter_sampler>0):
        send_iter=options.iter_sampler

        X[:,:,:,0]=X_np[:,:,:,send_iter-1]
        X[:,:,:,1]=X_np[:,:,:,send_iter-1]-X_np[:,:,:,send_iter-2]
        
    else:

        for i in range(l):
            send_iter=random.randint(0, 99)

            X[i,:,:,0]=X_np[i,:,:,send_iter-1]
            X[i,:,:,1]=X_np[i,:,:,send_iter-1]-X_np[i,:,:,send_iter-2]

    model.train(X, y,EPOCHS=options.epochs,batch_size=options.batch_size)
    
    save_path=os.path.join(options.save_prefix, options.iter_sampler)
    save_path=os.path.join(save_path,save_prefix)
    model.save_weights(save_path)

    
    
    
    
    
    
    
    
    
    
    
    
    print('end')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    