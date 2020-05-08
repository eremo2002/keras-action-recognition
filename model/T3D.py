import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Conv3D, Dropout, AveragePooling3D, MaxPooling3D
from tensorflow.keras.layers import  Dense, Flatten, GlobalAveragePooling2D, GlobalAveragePooling3D
from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.applications import densenet

def _DenseLayer(prev_layer, growth_rate, bn_size, drop_rate):
    if prev_layer is None:
        # print('No Layer previous to Dense Layers!!')
        return None
    else:
        x = BatchNormalization()(prev_layer)
    x = Activation('relu')(x)
    x = Conv3D(filters=bn_size * growth_rate, kernel_size=1, strides=1, padding='same')(x)    

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=growth_rate, kernel_size=3, strides=1, padding='same')(x)
    
    return x


def _DenseBlock(prev_layer, num_layers, bn_size, growth_rate, drop_rate):
    x = prev_layer
    
    for i in range(num_layers):
        layer = _DenseLayer(x, growth_rate, bn_size, drop_rate)
        if layer is None:
            print('Dense Block not created as no previous layers found!!')
            return None
        else:    
            x = keras.layers.concatenate([x, layer])
            
    return x


def _Transition(prev_layer, num_output_features):

    x = BatchNormalization()(prev_layer)
    x = Activation('relu')(x)
    x = Conv3D(filters=num_output_features, kernel_size=(1, 1, 1), strides=1, use_bias=False, padding='same')(x)
    
    x = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    
    return x


def _TTL(prev_layer, temporal_size=[1, 3, 4]):
    
    b1 = BatchNormalization()(prev_layer)
    b1 = Activation('relu')(b1)
    b1 = Conv3D(128, kernel_size=(temporal_size[0], 1, 1), strides=1, use_bias=False, padding='same')(b1)

    b2 = BatchNormalization()(prev_layer)
    b2 = Activation('relu')(b2)
    b2 = Conv3D(128, kernel_size=(temporal_size[1], 3, 3), strides=1, use_bias=False, padding='same')(b2)

    b3 = BatchNormalization()(prev_layer)
    b3 = Activation('relu')(b3)
    b3 = Conv3D(128, kernel_size=(temporal_size[2], 3, 3), strides=1, use_bias=False, padding='same')(b3)

    # x = keras.layers.concatenate([b1, b2, b3], axis=1)
    x = keras.layers.concatenate([b1, b2, b3], axis=-1)
   
    return x


def DenseNet3D(input_shape, growth_rate=32, block_config=(6, 12, 24, 16),
               num_init_features=64, bn_size=4, drop_rate=0.6, num_classes=5):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    #-----------------------------------------------------------------
    inp_2d = (Input(shape=(input_shape[1],input_shape[2],input_shape[3]), name='2d_input'))
    pretrained_densenet = densenet.DenseNet169(include_top=False, input_shape=(input_shape[1],input_shape[2],input_shape[3]), input_tensor=inp_2d, weights='imagenet')    
    
    for layer in pretrained_densenet.layers:
        layer.trainable = False
    #-----------------------------------------------------------------

    # First convolution-----------------------
    inp_3d = (Input(shape=input_shape, name='3d_input'))
    print('inp_3d', inp_3d.shape)


    # need to check padding
    x = (Conv3D(num_init_features, kernel_size=(3, 7, 7), strides=(1, 2, 2), padding='same', use_bias=False))(inp_3d)
    print('after first conv', x.shape)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # need to check padding
    # x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    print('after maxpool3d ', x.shape)

    # Each denseblock
    # num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        # print('Pass', i)
        x = _DenseBlock(x, num_layers=num_layers,
                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        if i == 0:
            x = _TTL(x, temporal_size=(1, 3, 6))
            print('after first ttl ', x.shape)

            x = _Transition(x, num_output_features=int(x.shape[-1])//2)
            print('first transition ', x.shape)

        elif i != 0 and i != len(block_config) - 1:
            x = _TTL(x, temporal_size=(1, 3, 4))
            print('after ttl ', x.shape)

            x = _Transition(x, num_output_features=int(x.shape[-1])//2)
            print('after transition ', x.shape)


        # if i != len(block_config) - 1:
        #     # print('Not Last layer, so adding Temporal Transition Layer')
        #     print('before ttl', x.shape)
        #     x = _TTL(x)
        #     print('after ttl', x.shape)
        #     # num_features = 128*3
            
        #     print('before transition', x.shape)
        #     x = _Transition(x, num_output_features=int(x.shape[-1])//2)
        #     print('after transition', x.shape)
        #   # num_features = num_features



    # Final batch norm
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    # x = GlobalAveragePooling3D()(x)
    x = AveragePooling3D(pool_size=(1, 7, 7))(x)
    x = Flatten(name='flatten_3d')(x)
    x = Dense(1024, activation='relu')(x)    
    
    #--------------fron 2d densenet model-----------------
    y = GlobalAveragePooling2D(name='avg_pool_densnet2d')(pretrained_densenet.output)
    y = Dense(1024, activation='relu')(y)

    #-----------------------------------------------------
    x = keras.layers.concatenate([x,y])
    x = Dropout(0.65)(x)    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.35)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[inp_2d, inp_3d], outputs=[out])
    
    return model


# the below model has the lowest Top-1 error in ImageNet Data Set:
def densenet169_3D_DropOut(input_shape, nb_classes):
    model = DenseNet3D(input_shape, growth_rate=48, block_config=(
        6, 12, 32, 32), num_init_features=96, drop_rate=0.6, num_classes=nb_classes)
    return model


def densenet121_3D_DropOut(input_shape, nb_classes):
    """Constructs a DenseNet-121_DropOut model.
    """
    model = DenseNet3D(input_shape, num_init_features=64, growth_rate=32,
                       block_config=(6, 12, 24, 16), drop_rate=0.6, num_classes=nb_classes)
    
    return model
