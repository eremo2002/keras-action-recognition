import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Conv3D, Dropout, Concatenate, AveragePooling3D, MaxPooling3D, Dense, Flatten, GlobalAveragePooling2D, GlobalAveragePooling3D, ZeroPadding2D
from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.applications import densenet, nasnet


"""
expand 2D-ResNet50 to 3D-ResNet50

I referenced 2D-ResNet architecture fchollet's keras code(below link)
https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


"""



def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=4, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=4, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=4, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=4, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=4, name=bn_name_base + '2c')(x)

    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=4, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=None, classes=1000):    
   
    input_3d = (Input(shape=input_shape, name='3d_input'))

    x = Conv3D(64, (3, 7, 7), strides=(2, 2, 2), padding='same', name='conv1')(inp_3d)
    x = BatchNormalization(axis=4, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1, 1))
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 256], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 512], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 1024], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 1024], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 1024], stage=5, block='c')

    # x = AveragePooling3D((1, 7, 7), name='avg_pool')(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling3D()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax', name='fc')(x)
   

    # Create model.
    model = Model(input_3d, x, name='resnet50')
    model.summary()
        
    return model


if __name__ == '__main__':
    model = ResNet50(input_shape=(16, 224, 224, 3), classes=1000)
