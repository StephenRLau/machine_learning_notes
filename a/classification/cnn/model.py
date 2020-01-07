import numpy as np
import keras
import data_util
from keras.models import Sequential
from keras.layers import Reshape
import json
from keras import backend as K
from keras.models import Model, Input
import tensorflow as tf

def cnn_model():
    model = Sequential()
    model.add( keras.layers.Conv2D( filters=30 , kernel_size=(5,5) , strides=(1,1) , padding='valid', activation='relu' , name='Conv1' ) )
    model.add( keras.layers.AvgPool2D( pool_size=(2,2) , padding='valid' , name='Avg1' ) )
    model.add( keras.layers.Conv2D( filters=30 , kernel_size=(5,5) , strides=(1,1) , padding='valid', activation='relu' , name='Conv2' ) )
    model.add( keras.layers.AvgPool2D( pool_size=(2,2) , padding='valid' , name='Avg2' ) )
    model.add( keras.layers.Conv2D( filters=30 , kernel_size=(5,5) , strides=(1,1) , padding='valid', activation='relu' , name='Conv3' ) )
    model.add( keras.layers.AvgPool2D( pool_size=(2,2) , padding='valid' , name='Avg3' ) )
    model.add( keras.layers.Flatten() )
    model.add( keras.layers.Dense( 10 , activation='softmax' ) )

    model.compile( optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics= [ 'accuracy' ] )

    return model

feature , label = data_util.csv_to_nparray()

print( feature.shape )
model = cnn_model()
model.fit( feature , label , batch_size = 64 , epochs=10 )