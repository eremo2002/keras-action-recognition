import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random

'''
this generator returns 'single image' and '8 frmaes clip'
'''

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, fpv, height, width, channels, num_classes, batch_size, shuffle=True):   
        self.dataframe = pd.read_csv(dataframe, index_col=False)
        self.fpv = fpv
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):        
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]        
#         print('indexes ', indexes)

        images = []
        clip = []
        labels = []

        for i in indexes:        
            input_2d, input_3d, y = self.get_data(i)
        
            images.append(input_2d)
            clip.append(input_3d)
            labels.append(y)

        # temp_2d = np.array(images)
        # temp_3d = np.array(clip)
        # temp_label = np.array(labels)
        # print(temp_2d.shape, temp_3d.shape, temp_label.shape)
        return (np.array(images), np.array(clip)), np.array(labels)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_data(self, idx):
        video = self.dataframe['path'].values[idx]        
        action_class = self.dataframe['class'].values[idx]

        cap = cv2.VideoCapture(video)

        frames = []

        while(True):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.height, self.width))
            frames.append(frame)
        cap.release()

        
        rnd_idx = random.randint(1, len(frames)-1)
        rnd_frame = frames[rnd_idx]
        
        step = len(frames)//self.fpv
        frames = frames[::step]
        frames = frames[:self.fpv]

#         for i in range(0, 8):                                        
#                     cv2.imshow('single clip frmae', frames[i])
#                     cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
        y = to_categorical(action_class, num_classes=self.num_classes)
        
        return rnd_frame, frames, y

        

