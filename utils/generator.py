import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
#import imgaug.augmenters as iaa

'''
Frame_Clip_DataGenerator returns 'single frame', 'video clip', 'label'
Frame_Flow_DataGenerator returns 'single frame', 'optical flow', 'label'
Clip_DataGenerator returns 'video clip', 'label'
'''


class Frame_Clip_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, fpv, height, width, channels, num_classes, frame_step, batch_size, shuffle=True):   
        self.dataframe = pd.read_csv(dataframe, index_col=False)        
        self.fpv = fpv
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.frame_step = frame_step
        self.batch_size = batch_size        
        self.shuffle = shuffle
        self.on_epoch_end()        

    def __len__(self):        
        return int(np.floor(len(self.dataframe) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]        
        
        x_list = [self.dataframe['path'].values[k] for k in indexes]
        y_list = [self.dataframe['class'].values[k] for k in indexes]

        img, clip, y = self.get_data(x_list, y_list)

        return (img, clip), y    


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def __iter__(self):
        return self


    def __next__(self):
        return self.next()


    def get_data(self, x_list, y_list):

        img = np.empty((self.batch_size, self.height, self.width, self.channels))
        clip = np.empty((self.batch_size, self.fpv, self.height, self.width, self.channels))
        y = np.empty((self.batch_size))

        for i in range (0, self.batch_size):
            
            cap = cv2.VideoCapture(x_list[i])
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                frame = cv2.resize(frame, (self.height, self.width))
                frames.append(frame)
            
            cap.release()

            rnd_init_index = random.randint(0, len(frames)//2)
            final_index = (self.fpv * self.frame_step) + rnd_init_index
            frames = frames[rnd_init_index:final_index:(self.frame_step)]
            
            img[i] = frames[-1]
            clip[i] = frames            
            
        y = to_categorical(y_list, num_classes=self.num_classes)

        return img/255.0, clip/255.0, y
  

        
class Frame_Flow_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, fpv, height, width, channels, num_classes, frame_step, batch_size, shuffle=True):   
        self.dataframe = pd.read_csv(dataframe, index_col=False)        
        self.fpv = fpv
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.frame_step = frame_step
        self.batch_size = batch_size        
        self.shuffle = shuffle
        self.on_epoch_end()        

    def __len__(self):        
        return int(np.floor(len(self.dataframe) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]        
        
        x_list = [self.dataframe['path'].values[k] for k in indexes]
        y_list = [self.dataframe['class'].values[k] for k in indexes]

        img, flow, y = self.get_data(x_list, y_list)

        return (img, flow), y    


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def __iter__(self):
        return self


    def __next__(self):
        return self.next()
        

    def get_data(self, x_list, y_list):

        img = np.empty((self.batch_size, self.height, self.width, self.channels))
        flows = np.empty((self.batch_size, self.fpv, self.height, self.width, self.channels))
        y = np.empty((self.batch_size))

        for i in range (0, self.batch_size):
            
            cap = cv2.VideoCapture(x_list[i])
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                frame = cv2.resize(frame, (self.height, self.width))
                frames.append(frame)
            
            cap.release()

            rnd_init_index = random.randint(0, len(frames)//2)
            final_index = (self.fpv * self.frame_step) + rnd_init_index
            frames = frames[rnd_init_index:(final_index+self.frame_step):self.frame_step]
            


            '''
            calculate optical flow
            '''
            flow_list = []
            
            prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frames[0])
            hsv[...,1] = 255

            j = 1
            while j < len(frames):
                next_frame = frames[j]
                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                optical_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)       
                
                flow_list.append(optical_flow)

                prev_frame = next_frame
                j = j+1
            
            # for i in range(0, self.fpv):                                        
            #     cv2.imshow('check', flow_list[i])
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img[i] = frames[-1]
            flows[i] = flow_list            
            
        y = to_categorical(y_list, num_classes=self.num_classes)

        return img/255.0, flows, y


class Clip_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, fpv, height, width, channels, num_classes, frame_step, batch_size, shuffle=True):   
        self.dataframe = pd.read_csv(dataframe, index_col=False)        
        self.fpv = fpv
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.frame_step = frame_step
        self.batch_size = batch_size        
        self.shuffle = shuffle
        self.on_epoch_end()        

    def __len__(self):        
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]        
        
        x_list = [self.dataframe['path'].values[k] for k in indexes]
        y_list = [self.dataframe['class'].values[k] for k in indexes]
        # print(x_list, y_list)

        clip, y = self.get_data(x_list, y_list)

        return clip, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_data(self, x_list, y_list):

        clip = np.empty((self.batch_size, self.fpv, self.height, self.width, self.channels))
        y = np.empty((self.batch_size))

        for i in range (0, self.batch_size):
            
            cap = cv2.VideoCapture(x_list[i])
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                frame = cv2.resize(frame, (self.height, self.width))
                frames.append(frame)
            
            cap.release()

            rnd_init_index = random.randint(0, len(frames)//2)
            final_index = (self.fpv * self.frame_step) + rnd_init_index
            frames = frames[rnd_init_index:final_index:(self.frame_step)]
            
            clip[i] = frames            
            
        y = to_categorical(y_list, num_classes=self.num_classes)

        return clip/255.0, y
