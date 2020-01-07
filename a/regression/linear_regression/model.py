import json
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np

def data_load():
    with open( './data/train.json' , 'r' , encoding='utf-8' ) as f:
        train_data = json.load(f)
    
    with open( './data/test.json' , 'r' , encoding='utf-8' ) as f:
        test_data = json.load(f)

    x_train = [ ]
    y_train = [ ]
    x_test = [ ]
    y_test = [ ]
    attribute = [ "AMB_TEMP" , "CH4" , "CO" , "NMHC" , "NO" , "NO2" , "NOx" , "O3" , "PM10" , "PM2.5" , "RAINFALL" , "RH" , "SO2" , "THC" , "WD_HR" , "WIND_DIREC" , "WIND_SPEED" , "WS_HR" ]
    for i in range( 0 , len(train_data) ):
        tmp_x_train = [ ]
        for j in range( 0 , len(attribute) ):
            if attribute[j] != 'PM2.5':
                if train_data[i][attribute[j]] == 'NR' :
                    tmp_x_train.append( 0 )
                else :
                    tmp_x_train.append( train_data[i][attribute[j]] )
            else :
                y_train.append( train_data[i][attribute[j]] )
        x_train.append( tmp_x_train )
    
    for i in range( 0 , len(test_data) ):
        tmp_x_test = [ ]
        for j in range( 0 , len(attribute) ):
            if attribute[j] != 'PM2.5':
                if test_data[i][attribute[j]] == 'NR' :
                    tmp_x_test.append( 0 )
                else :
                    tmp_x_test.append( test_data[i][attribute[j]] )
            else :
                y_test.append( test_data[i][attribute[j]] )
        x_test.append( tmp_x_test )
    
    x_train = np.array( x_train ,dtype='float64' )
    x_test = np.array( x_test , dtype='float64' )
    y_train = np.array( y_train , dtype='float64' )
    y_test = np.array( y_test , dtype='float64' )

    return x_train , y_train , x_test , y_test

def model( ):
    x_train , y_train , x_test , y_test = data_load()
    lrg = LinearRegression( normalize=True ) # 加上regularization
    lrg.fit( x_train , y_train )
    print( lrg.score( x_test , y_test ) )

if __name__ == "__main__":
    model( )

