import numpy as np
import keras
import data_util
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional , Average , Masking
from keras.datasets import imdb
from keras.layers import Reshape
import json
from keras import backend as K
from keras.models import Model, Input
import tensorflow as tf
import data_util
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0" 

def decode_sequence(input_seq , id_2_vocab , encoder_model , decoder_model ):

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 1.  # { 'PAD' : 0 ,  'ST1' : 1 , 'EN1' : 2 }

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict( [target_seq] + states_value )

        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = id_2_vocab[sampled_token_index]
        print( 'input:' , id_2_vocab[ int(target_seq[0][0]) ] , 'output:' , sampled_char )
        decoded_sentence = decoded_sentence + sampled_char + ' '
        
        if sampled_char == 'EN1' or len(decoded_sentence) > 100 :
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0 ] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

def translate_model2(  num_encoder_tokens , num_decoder_tokens , latent_dim , maxlen):
    encoder_inputs = Input(shape=(maxlen,))
    encoder_embedder = Embedding( num_encoder_tokens , 50 , mask_zero=True , name='src_embed' )
    encoder_embed = encoder_embedder( encoder_inputs )
    
    encoder = LSTM( 100 , return_state=True , name='src_lstm' )
    encoder_outputs, state_h, state_c = encoder(encoder_embed)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(maxlen,))
    decoder_embedder = Embedding( num_decoder_tokens , 50 , mask_zero=True , name='tgt_embed' )
    decoder_embed = decoder_embedder( decoder_inputs )
    
    decoder_lstm = LSTM( 100 , return_sequences=True, return_state=True , name='tgt_lstm' )
    decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax' , name='tgt_dense' )
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # opt = keras.optimizers.RMSprop(0.01)
    model.compile( optimizer='rmsprop' , loss='sparse_categorical_crossentropy' )
    # model.summary()
    
    
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(100,))
    decoder_state_input_c = Input(shape=(100,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs_1 = Input(shape=(1,))
    decoder_embed_1 = decoder_embedder( decoder_inputs_1 )
    decoder_outputs_1, state_h_1, state_c_1 = decoder_lstm(decoder_embed_1, initial_state=decoder_states_inputs)
    decoder_states = [state_h_1, state_c_1]
    decoder_outputs_1 = decoder_dense(decoder_outputs_1)

    decoder_model = Model( [decoder_inputs_1] + decoder_states_inputs , [decoder_outputs_1] + decoder_states )
    
    return model , encoder_model , decoder_model



def train():
    data_pre = data_util.data_util()
    chinese_sent_id , english_sent_id , cn_id_2_vocab , eng_id_2_vocab = data_pre.read_data( re_write_map=True )

    src_word_number = len( cn_id_2_vocab )
    tgt_word_number = len( eng_id_2_vocab )
    maxlen = 20
    hidden_size = 100
    batch_size = 128
    model , _ , _ = translate_model2( src_word_number , tgt_word_number , hidden_size , maxlen)

    max_cn = max( [ len(i) for i in chinese_sent_id ] )
    max_eng = max( [ len(i) for i in english_sent_id ] )
    print( max_cn , max_eng )

    train_src_input = chinese_sent_id[ : ]
    valid_src_input = chinese_sent_id[ : ]
    train_tgt_input = english_sent_id[ : ]
    valid_tgt_input = english_sent_id[ : ]

    train_tgt_output = []
    valid_tgt_output = []
    for i in range( 0 , len(train_tgt_input) ):
        train_tgt_output.append( train_tgt_input[i][1:] )
    
    for i in range( 0 , len(valid_tgt_input) ):
        valid_tgt_output.append( valid_tgt_input[i][1:] )

    train_src_input = sequence.pad_sequences(train_src_input, maxlen=maxlen , padding='post' , truncating='post')
    valid_src_input = sequence.pad_sequences(valid_src_input, maxlen=maxlen , padding='post' , truncating='post')
    train_tgt_input = sequence.pad_sequences(train_tgt_input, maxlen=maxlen , padding='post' , truncating='post')
    valid_tgt_input = sequence.pad_sequences(valid_tgt_input, maxlen=maxlen , padding='post' , truncating='post')
    train_tgt_output = sequence.pad_sequences(train_tgt_output, maxlen=maxlen , padding='post' , truncating='post')
    valid_tgt_output = sequence.pad_sequences(valid_tgt_output, maxlen=maxlen , padding='post' , truncating='post')

    y_train = np.expand_dims( train_tgt_output ,-1)
    y_test = np.expand_dims( valid_tgt_output ,-1)

    print( "input" , train_src_input[0] )
    print( "output" , train_tgt_input[0] )
    print( y_train[0] )

    src_sent = ''
    for i in range( 0 , len(train_src_input[1])):
        src_sent = src_sent + cn_id_2_vocab[ train_src_input[1][i] ] + ' '
    print( "src_sent:" , src_sent , '\n')

    tgt_sent_output = ''
    for i in range( 0 , len(train_tgt_output[1])):
        tgt_sent_output = tgt_sent_output + eng_id_2_vocab[ train_tgt_output[1][i] ] + ' '
    print( "tgt_sent_output" , tgt_sent_output , '\n')
    
    model.fit([train_src_input, train_tgt_input], y_train , batch_size=batch_size, epochs=250 , validation_data=[ [valid_src_input, valid_tgt_input], y_test ])
    model.save_weights('./weight/checkpoint.hdf5')
    


def test():
    data_pre = data_util.data_util()
    chinese_sent_id , english_sent_id , cn_id_2_vocab , eng_id_2_vocab = data_pre.read_data( re_write_map=False )

    src_word_number = len( cn_id_2_vocab )
    tgt_word_number = len( eng_id_2_vocab )
    maxlen = 20
    hidden_size = 100
    model , encoder_model , decoder_model = translate_model2( src_word_number , tgt_word_number , hidden_size , maxlen)
    train_src_input = chinese_sent_id[ : ]
    train_tgt_input = english_sent_id[ : ]

    train_tgt_output = []
    valid_tgt_output = []
    for i in range( 0 , len(train_tgt_input) ):
        train_tgt_output.append( train_tgt_input[i][1:] )
    
    train_src_input = sequence.pad_sequences(train_src_input, maxlen=maxlen , padding='post' , truncating='post')
    train_tgt_input = sequence.pad_sequences(train_tgt_input, maxlen=maxlen , padding='post' , truncating='post')
    train_tgt_output = sequence.pad_sequences(train_tgt_output, maxlen=maxlen , padding='post' , truncating='post')
    valid_tgt_output = sequence.pad_sequences(valid_tgt_output, maxlen=maxlen , padding='post' , truncating='post')

    model.load_weights('./weight/checkpoint.hdf5')
    model.summary()
    encoder_model.summary()
    decoder_model.summary()
    
    # print( train_src_input[1:2] )
    # print( train_tgt_input[1:2] )

    sents = decode_sequence( train_src_input[1:2] , eng_id_2_vocab , encoder_model , decoder_model )
    src_sent = ''
    for i in range( 0 , len(train_src_input[1])):
        src_sent = src_sent + cn_id_2_vocab[ train_src_input[1][i] ] + ' '
    
    
    print( "src_sent:" , src_sent , '\n')
    tgt_sent = ''
    for i in range( 0 , len(train_tgt_input[1])):
        tgt_sent = tgt_sent + eng_id_2_vocab[ train_tgt_input[1][i] ] + ' '
    print( "tgt_sent_input:" , tgt_sent , '\n' )
    print( "sents:" , sents  ,'\n')
    
    answers = model.predict( [ train_src_input[1:2] , train_tgt_input[1:2] ] )
    
    tgt_sent_output = ''
    for i in range( 0 , len(train_tgt_output[1])):
        tgt_sent_output = tgt_sent_output + eng_id_2_vocab[ train_tgt_output[1][i] ] + ' '
    print( "tgt_sent_output" , tgt_sent_output , '\n')

    for i in range( 0 , len(answers[0]) ):
        print("input:" ,  eng_id_2_vocab[ train_tgt_input[1][i] ] , "output:" , eng_id_2_vocab[ np.argmax(answers[0, i, :]) ] )
    # print( y_train[1:2])
    


# train()
test()
# 一共2大模块
# 1. encoder
# 2. decoder
# 
# train的时候是通过 input 和 target一起进行的
# 
