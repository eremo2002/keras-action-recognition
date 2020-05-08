import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Dense, Dropout, MaxPool3D, Flatten, Activation
from tensorflow.keras.models import Sequential, Model


def C3D_model(input_shape, nb_classes):
    inp_3d = Input(shape=(input_shape), name='3d_input')

    x = Conv3D(64, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(inp_3d)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(x)
    print(x.shape)

    x = Conv3D(128, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    print(x.shape)

    x = Conv3D(256, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = Conv3D(256, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    print(x.shape)

    x = Conv3D(512, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = Conv3D(512, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    print(x.shape)

    x = Conv3D(512, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = Conv3D(512, kernel_size=(3, 3, 3),strides=(1, 1, 1), padding='same',activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    print(x.shape)

    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = Model(inp_3d, x)
    return model
