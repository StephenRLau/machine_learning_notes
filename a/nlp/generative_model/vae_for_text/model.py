import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input , Embedding , LSTM
from keras.preprocessing import sequence
import json
import data_util
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0" 

def autoencoder( maxlen ,word_number ):
    encoder_input = Input( shape=( maxlen, ) )
    encoder_embedding_layer = Embedding( word_number , 50 , mask_zero = True , name='encoder_embedding_layer' )
    encoder_embeded = encoder_embedding_layer( encoder_input )
    encoder_lstm = LSTM( 50 , return_state=True , name='encoder_lstm' )
    encoder_output , state_h , state_c = encoder_lstm( encoder_embeded )

    encoder_state = [ state_h , state_c ]
    
    decoder_input = Input( shape=( maxlen , ) )
    decoder_embedding_layer = Embedding( word_number , 50 , mask_zero = True , name='decoder_embedding_layer' )
    decoder_embeded = decoder_embedding_layer( decoder_input )
    decoder_lstm = LSTM( 50 , return_sequences=True, return_state=True , name='decoder_lstm' )
    decodered , _ , _ = decoder_lstm( decoder_embeded , initial_state=encoder_state )
    decoder_dense = Dense( word_number , activation='softmax' , name='decoder_dense' )

    output = decoder_dense( decodered )


    # optimizer = keras.optimizers.rmsprop(0.001)
    model = Model( [encoder_input , decoder_input] , output )
    model.compile( optimizer='Adam' , loss='sparse_categorical_crossentropy' )

    model.summary() 
    return model


def train(  ):
    epochs = 100
    batch_size = 64
    data_pro = data_util.data_util( )
    [ input_sent_ids , target_input_sent_ids , target_output_sent_ids ] , char_dict , char_id_2_dict = data_pro.read_data()
    
    print(len(char_dict))
    max_length = max( [ len(i) for i in target_input_sent_ids ] )
    
    input_sent_ids = sequence.pad_sequences( input_sent_ids , maxlen=max_length , padding='post' , truncating='post' )
    target_input_sent_ids = sequence.pad_sequences( target_input_sent_ids , maxlen=max_length , padding='post' , truncating='post' )
    target_output_sent_ids = sequence.pad_sequences( target_output_sent_ids , maxlen=max_length , padding='post' , truncating='post' )

    y_target = np.expand_dims( target_output_sent_ids ,-1)

    print( input_sent_ids[0] )
    print( target_input_sent_ids[0] )
    print( target_output_sent_ids[0] )
    
    model = autoencoder( max_length , len(char_dict) )
    # model.load_weights( './weight/checkpoint.hdf5' )
    # print('load')
    model.fit( [ input_sent_ids , target_input_sent_ids ] , y_target , batch_size=batch_size , epochs=epochs )
    model.save_weights('./weight/checkpoint.hdf5')
    

if __name__ == "__main__":
    train()


# init_state = 1

