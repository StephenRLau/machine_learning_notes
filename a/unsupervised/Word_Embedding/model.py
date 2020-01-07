import tensorflow as tf
import keras
from keras.models import Sequential

def lstm_model( ):
    model = Sequential()
    model.add( keras.layers.LSTM() )
