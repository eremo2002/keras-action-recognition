# import c
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Conv3D, Dropout, Concatenate, AveragePooling3D, MaxPooling3D, Dense, Flatten, GlobalAveragePooling2D, GlobalAveragePooling3D, ZeroPadding2D
from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.applications import densenet, nasnet






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
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
   
    inp_3d = (Input(shape=input_shape, name='3d_input'))

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
    model = Model(inp_3d, x, name='resnet50')
    model.summary()
        
    return model


if __name__ == '__main__':
    model = ResNet50(input_shape=(16, 224, 224, 3), classes=50)
    model.summary()
