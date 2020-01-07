import csv
import json


def csv_to_json( sFileName='./data/train.csv' , output_filename = './data/train.json' ):
    json_data = [ ]
    with open( sFileName , newline='', encoding='big5') as csvfile:
        rows=csv.reader(csvfile)
        day = ''
        for row in rows:
            if row[0] == '日期':
                tmp_json_data = []
                continue
            if day != row[0]:
                if len(row) == 27 :
                    length = len(row) - 3
                    attribute_index = 2
                else :
                    length = len(row) - 2
                    attribute_index = 1
                if 'tmp_json_data' in locals().keys() and len( tmp_json_data ) > 0 :
                    json_data.extend( tmp_json_data )
                tmp_json_data = []
                for i in range( 0 , length ):
                    tmp_json_data.append( { 'day' : row[0] , 'time' : i , row[attribute_index] : row[i+attribute_index+1] } )
                day = row[0]
            else :
                if len(row) == 27 :
                    length = len(row) - 3
                    attribute_index = 2
                else :
                    length = len(row) - 2
                    attribute_index = 1
                for i in range( 0 , length ):
                    tmp_json_data[i][row[attribute_index]] = row[i+attribute_index+1]
        
        json_data.extend( tmp_json_data )
    
    with open( output_filename , 'w' , encoding='utf-8' ) as f :
        json.dump( json_data , f , ensure_ascii=False , indent=1 )

if __name__ == "__main__":
    csv_to_json()
    csv_to_json('./data/test.csv' , './data/test.json')