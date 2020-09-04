import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import traceback

from model.twostream_3dcnn import twostream_model
from utils.result_graph import plot_history, save_history
from utils.rgb_flow_generator import RGB_Flow_DataGenerator

K.clear_session()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# there is a minimum number of frames that the network must have, values below 10 gives -- ValueError: Negative dimension size caused by subtracting 3 from 2 for 'conv3d_7/convolution'
# paper uses 224x224, but in that case also the above error occurs
FRAMES_PER_VIDEO = 16
FRAME_HEIGHT = 300
FRAME_WIDTH = 300
FRAME_CHANNEL = 3
FRAME_STEP = 2
BATCH_SIZE = 12
EPOCHS = 100


sample_input = np.empty([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

# Read Dataset
d_test = pd.read_csv(os.path.join('test_data.csv'))
nb_classes = len(set(d_test['class']))

print(f'testdatasets {len(d_test)}')
print(f'classes {nb_classes}')



# video_test_generator = RGB_Flow_DataGenerator('test_data.csv', 
#                                             'test',
#                                             FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
#                                             4, 
#                                             FRAME_STEP, 
#                                             batch_size=BATCH_SIZE, 
#                                             shuffle=False)


# from result4_C3D_clip_flow_crop_fpv16_wh300_step2.twostream_C3D import twostream_model
# model = twostream_model([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], 4)   
# model.load_weights('result4_C3D_clip_flow_crop_fpv16_wh300_step2/weights_epoch-086_acc-0.9937_valloss-0.0274_valacc-0.9900.h5')
# model.summary()






from utils.C3D_generator import Clip_Generator

video_test_generator = Clip_Generator('test_data.csv', 
                                            'test',
                                            FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, 
                                            4, 
                                            FRAME_STEP, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=False)

from result3_C3D_clip_crop_fpv16_wh300_step2.C3D import C3D_model
model = C3D_model([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], 4)   
model.load_weights('result3_C3D_clip_crop_fpv16_wh300_step2/weights_epoch-086_acc-0.9831_valloss-0.1105_valacc-0.9949.h5')
model.summary








model.compile(optimizer=Adam(lr=1e-4, decay=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])    

print('Training started....')    

train_steps = len(d_test)//BATCH_SIZE

print(f'train_steps {train_steps}')

scores = model.evaluate_generator(video_test_generator, 
                                    workers=15,
                                    use_multiprocessing=True)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# scores = model.predict_generator(video_test_generator, 
#                                     workers=15,
#                                     use_multiprocessing=True)
# print(scores)

