import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import cv2


def load_model():
    model , enc , dec = autoencoder_model()
    model.load_weights('./weight/weights_v2.hdf5')
    return model , enc , dec

def encoder():
    input_img = Input(shape=(784,) , name='input_1') # 编码层
    # input_decoder = Input(shape=(2,) , name='input_2' )

    encoded = Dense(128, activation='relu' , name='dense_1' )(input_img)
    encoded = Dense(64, activation='relu' ,  name='dense_2' )(encoded)
    encoded = Dense(10, activation='relu' ,  name='dense_3' )(encoded)
    encoder_output = Dense(2 , name='decoder_input')(encoded) # 解码层

    encoder = Model(inputs=input_img , outputs=encoder_output )
    return encoder

def decoder():
    input_code = Input( shape=(2,) , name='input_decoder' )
    decoded_0 = Dense(10, activation='relu' , name='dense_4' )(input_code)
    decoded_1 = Dense(64, activation='relu' , name='dense_5' )(decoded_0)
    decoded_2 = Dense(128, activation='relu', name='dense_6' )(decoded_1)
    decoded = Dense(784, activation='tanh' , name='dense_output' )(decoded_2) # 构建自编码模型

    decoder = Model(inputs=input_code , outputs=decoded  )

    return decoder
    

def autoencoder_model():
    input_img = Input(shape=(784,) , name='input_1')
    enc = encoder()
    dec = decoder()
    
    enc_out = enc(input_img)
    dec_out = dec(enc_out)

    autoencoder = Model(inputs=input_img, outputs=dec_out) # 构建编码模型
    autoencoder.compile(optimizer='adam', loss='mse') # training

    return autoencoder , enc , dec


def load_data():
    (x_train, _), (x_test, y_test) = mnist.load_data() # 数据预处理
    x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
    x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train , x_test

def train():
    x_train , x_test = load_data()
    autoencoder , enc , dec= autoencoder_model()
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True) # plotting
    autoencoder.save_weights( './weight/weights_v2.hdf5' )

def test():
    x_train , x_test = load_data()
    autoencoder , enc , dec= autoencoder_model()
    autoencoder.load_weights( './weight/weights_v2.hdf5' )
    # output_1 = enc.predict( x_train[:1] )
    tmp_input = np.random.uniform(1, 10, (1, 2))
    print(  tmp_input  )
    output_2 = dec.predict( tmp_input )

    output_2 = (output_2 + 0.5) * 255 
    output_2 = output_2.astype('int32')
    output_2 = output_2[0]
    for i in range( 0 , len(output_2) ):
        if output_2[i] < 0 :
            output_2[i] = 0
    
    output_2 = output_2.reshape( 28, 28 )
    cv2.imwrite( 'image.jpg' , output_2 )
    # print( output_2 )

if __name__ == "__main__":
    # train()
    test()