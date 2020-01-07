import pandas as pd
import numpy as np
from tqdm import trange
import cv2
import os

def csv_to_images( ):
    name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    train_df = pd.read_csv("./data/train.csv")

    label = train_df['label']
    feature = train_df['feature'].map(lambda x : np.array(list(map(float, x.split()))))

    for i in trange( 0 , len(label) ):
        label_name = name[ label[i] ]
        data = feature[i]
        data = data.reshape( 48 , 48 )
        if not os.path.exists('./train_data/'+ label_name):
            os.mkdir( './train_data/'+ label_name )
        cv2.imwrite( './train_data/'+ label_name+'/'+ str(i) +'.jpg' , data )
    
    test_df = pd.read_csv("./data/test.csv")

    feature = test_df['feature'].map(lambda x : np.array(list(map(float, x.split()))))

    for i in trange( 0 , len(feature) ):
        data = feature[i]
        data = data.reshape( 48 , 48 )
        cv2.imwrite( './test_data/'+ str(i) +'.jpg' , data )

def csv_to_nparray( ):
    train_df = pd.read_csv("./data/train.csv")
    label = train_df['label']
    feature = train_df['feature'].map(lambda x : np.array(list(map(float, x.split()))))
    data = []
    for i in range( 0 , len(feature) ):
        data.append( feature[i].reshape( 48 , 48 , 1 ) )
    feature = np.array( data )
    label = np.array( label )
    return feature , label

# csv_to_images( )
# 0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立
# angry	disgust	fear	happy	sad	surprise	neutral