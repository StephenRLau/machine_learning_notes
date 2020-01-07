import json

class data_util(object):
    def __init__(self , re_write = False ):
        self.re_write = re_write

    def get_char_dict( self , sents):
        char_dict = { 'PAD':0 , 'UNK':1 , 'ST1':2 , 'EN1':3 }
        char_id_2_dict = [ 'PAD' , 'UNK' , 'ST1' , 'EN1' ]
        char_dict_list = [ ]
        for i in range( 0 , len(sents) ):
            char_list = [ sents[i][k] for k in range( 0 , len(sents[i])) ]
            #print(char_list)
            char_dict_list.extend( char_list )

        char_dict_set = set( char_dict_list )

        index = 4
        
        for key in char_dict_set:
            char_dict[ key ] = index
            index = index + 1
            char_id_2_dict.append(key)
        
        return char_dict , char_id_2_dict

    def sent_2_id( self, char_dict , sent ):
        sent_id = [ ]
        for i in range( 0 , len(sent) ):
            if sent[i] in char_dict:
                sent_id.append( char_dict[sent[i]] )
            else :
                sent_id.append( 1 )
        return sent_id

    def read_data( self ):
        with open( './dataset/poet_song.json' , 'r' , encoding='utf-8' ) as f:
            data = json.load(f)
        
        sents = []
        for i in range( 0 , len(data) ):
            # tmp_sent = ''
            if len(data[i]['paragraphs']) > 2:
                continue
            for j in range( 0 ,int(len(data[i]['paragraphs'])/2) ):
                tmp_sent = ''
                tmp_sent = tmp_sent + data[i]['paragraphs'][2*j] + data[i]['paragraphs'][2*j + 1 ]
                # print( tmp_sent )
                sents.append(tmp_sent)
        
        maxlen = max( [ len(key) for key in sents ] )
        # print(maxlen)
        if self.re_write:
            char_dict , char_id_2_dict = self.get_char_dict(sents)
            with open( './dataset/char_dict.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( char_dict , f , ensure_ascii=False , indent=1 )
            with open( './dataset/char_id_2_dict.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( char_id_2_dict , f , ensure_ascii=False , indent=1 )
            
        else :
            with open( './dataset/char_dict.json' , 'r' , encoding='utf-8' ) as f:
                char_dict = json.load(f)
            with open( './dataset/char_id_2_dict.json' , 'r' , encoding='utf-8' ) as f:
                char_id_2_dict = json.load(f)

        input_sent_ids = []
        target_input_sent_ids = []
        target_output_sent_ids = []
        for i in range( 0 , len(sents) ):
            tmp_sent_2_id = self.sent_2_id(char_dict , sents[i])
            input_sent_ids.append( tmp_sent_2_id )
            target_input_sent_ids.append( [2] + tmp_sent_2_id + [3] )
            target_output_sent_ids.append( tmp_sent_2_id + [3] )
        
        # print(len(sents))
        return [ input_sent_ids , target_input_sent_ids , target_output_sent_ids ] , char_dict , char_id_2_dict

# data_util().read_data()
