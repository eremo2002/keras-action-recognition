import os
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from model.C3D import C3D_model
from model.T3D import densenet169_3D_DropOut, densenet121_3D_DropOut, DenseNet3D
from model.SlowFast import SlowFast_Network

from utils.generator import Frame_Clip_DataGenerator, Clip_DataGenerator, Frame_Flow_DataGenerator
from utils.result_graph import plot_history, save_history



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


FRAMES_PER_VIDEO = 8
FRAME_HEIGHT = 256
FRAME_WIDTH = 256
FRAME_CHANNEL = 3
FRAME_STEP = 4
BATCH_SIZE = 8
EPOCHS = 200
result_path = 'result/'


def train():
    parser = argparse.ArgumentParser(description='argparse argument')
    parser.add_argument('--input', '-input',
                        help='frame_clip = single frame & video clip,   clip=video clip,   frame_flow=single frame & optical flow',
                        default='frame_clip', 
                        dest='input')
    parser.add_argument('--model', '-model', 
                        help='choose action recognition model', 
                        default='T3D-densenet121', 
                        dest='model')
    args = parser.parse_args()

    input_tensor = np.empty([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)



    d_train = pd.read_csv(os.path.join('train.csv'))
    d_valid = pd.read_csv(os.path.join('val.csv'))
    nb_classes = len(set(d_train['class']))



    T3D_support = ('frame_clip', 'frame_flow')
    C3D_support = ('clip')
    SlowFast_support = ('clip')


    if args.input == 'frame_clip':
        video_train_generator = Frame_Clip_DataGenerator('train.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True)
        
        video_val_generator = Frame_Clip_DataGenerator('val.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=False)   
    elif args.input == 'clip':
        video_train_generator = Clip_DataGenerator('train.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True)
        
        video_val_generator = Clip_DataGenerator('val.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=False)   
    elif args.input == 'frame_flow':
        video_train_generator = Frame_Flow_DataGenerator('train.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True)
        
        video_val_generator = Frame_Flow_DataGenerator('val.csv', 
                                                    FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                                    nb_classes, 
                                                    FRAME_STEP, 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=False)    



    if args.model == 'T3D-densenet121':        
        if args.input not in T3D_support:
            print('Use --input frame_clip or frame_flow')
            return
        model = densenet121_3D_DropOut(input_tensor.shape, nb_classes)
    elif args.model == 'T3D-densenet169':
        if args.input not in T3D_support:
            print('Use --input frame_clip or frame_flow')
            return
        model = densenet169_3D_DropOut(input_tensor.shape, nb_classes)
    elif args.model == 'C3D':
        if args.input not in C3D_support:
                print('C3D only support --input clip')
                return
        model = C3D_model(input_tensor.shape, nb_classes)
    elif args.model == 'SlowFast':
        if args.input not in SlowFast_support:
            print('SlowFast only support --input clip')
            return
        model = SlowFast_Network(input_tensor.shape, nb_classes)

   
     # model = multi_gpu_model(model, gpus=2)

   
    checkpoint = ModelCheckpoint(result_path+'weights_epoch-{epoch:03d}_acc-{acc:.4f}_valloss-{val_loss:.4f}_valacc-{val_acc:.4f}.h5', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True,
                                    mode='min', 
                                    save_weights_only=True)   

    earlyStop = EarlyStopping(monitor='val_loss',
                                mode='min', 
                                patience=30)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                        factor=0.1,
                                        patience=15, 
                                        verbose=1, 
                                        mode='min', 
                                        min_delta=0.0001, 
                                        cooldown=2, 
                                        min_lr=1e-6)    

    
    callbacks_list = []
    
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])    


    train_steps = len(d_train)//BATCH_SIZE
    val_steps = len(d_valid)//BATCH_SIZE
    
    history = model.fit_generator(video_train_generator,
                                    steps_per_epoch=train_steps,
                                    epochs=EPOCHS,
                                    validation_data=video_val_generator,
                                    validation_steps=val_steps,
                                    verbose=1,
                                    callbacks=callbacks_list,        
                                    workers=1,        
                                    use_multiprocessing=False)

    plot_history(history, result_path)
    save_history(history, result_path)


if __name__ == '__main__':
    train()    
    K.clear_session()
