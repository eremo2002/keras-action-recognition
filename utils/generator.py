import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import imgaug.augmenters as iaa

'''
Frame_Clip_DataGenerator returns 'single frame', 'video clip', 'label'
Clip_DataGenerator returns 'video clip', 'label'
Frame_Flow_DataGenerator returns 'single frame', 'optical flow', 'label'
Clip_Flow_DataGenerator returns 'video clip', 'optical flow', 'label'

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
        # print('indexes ', indexes)

        images = []
        clip = []
        labels = []
        
        for i in indexes:        
            input_2d, input_3d, y = self.get_data(i)

            images.append(input_2d)
            clip.append(input_3d)
            labels.append(y)
        return (np.array(images)/255.0, np.array(clip)/255.0), np.array(labels)
        

        


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

        # for i, _ in enumerate(frames):
        #     frames[i] = cv2.resize(frames[i], (self.height, self.width))
        
        # for i in range(0, 8):                                        
        #             cv2.imshow('single clip frmae', frames[i])
        #             cv2.waitKey(0)
        # cv2.destroyAllWindows()

        y = to_categorical(action_class, num_classes=self.num_classes)
        
        return frames[-1], frames, y

        
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
        
        clip = []
        labels = []
        
        for i in indexes:        
            input_3d, y = self.get_data(i)

            
            clip.append(input_3d)
            labels.append(y)
        return (np.array(clip)/255.0), np.array(labels)
        


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

        y = to_categorical(action_class, num_classes=self.num_classes)
        
        return frames, y

        
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
        
        images = []
        flow = []
        labels = []
        
        for i in indexes:        
            input_2d, input_3d, y = self.get_data(i)
            
            images.append(input_2d)
            flow.append(input_3d)
            labels.append(y)
        return (np.array(images)/255.0, np.array(flow)/255.0), np.array(labels)
        


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

        i = 1
        while i < len(frames):
            next_frame = frames[i]
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            optical_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)       
            
            flow_list.append(optical_flow)

            prev_frame = next_frame
            i = i+1

        
        # for i in range(0, 8):                                        
        #     cv2.imshow('check', flow_list[i])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        y = to_categorical(action_class, num_classes=self.num_classes)
        
        return frames[-1], flow_list, y


class Clip_Flow_DataGenerator(tf.keras.utils.Sequence):
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
        
        clip = []
        flow = []
        labels = []
        
        for i in indexes:        
            input_clip, input_flow, y = self.get_data(i)
            
            clip.append(input_clip)
            flow.append(input_flow)
            labels.append(y)
        return (np.array(clip)/255.0, np.array(flow)/255.0), np.array(labels)
        


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

        i = 1
        while i < len(frames):
            next_frame = frames[i]
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            optical_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)       
            
            flow_list.append(optical_flow)

            prev_frame = next_frame
            i = i+1

        
        # for i in range(0, 8):                                        
        #     cv2.imshow('check1', frames[i])
        #     cv2.imshow('check2', flow_list[i])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        y = to_categorical(action_class, num_classes=self.num_classes)
        
        return frames, flow_list, y

