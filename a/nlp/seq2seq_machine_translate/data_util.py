"""
    数据预处理程序
"""

import json
import re
# import jieba
import random
from keras.preprocessing import sequence
from langconv import *

# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


class data_util(object):
    def __init__( self , source_language='cn' ,target_language='eng'):
        self.source_language = source_language
        self.target_language = target_language

    def make_eng_vocab_dict( self , sent_list ):
        """
            sent_list : 为输入的句子list
            生成 词典数据
        """
        id_2_vocab = [ 'PAD' , 'ST1' , 'EN1' ]
        vocab_dict = { 'PAD' : 0 , 'ST1' : 1 , 'EN1' : 2 }
        vocab_index = 3
        new_vocab = [ ]
        for sent in sent_list :
            sent = sent.lower()
            tmp_vocab = re.sub('([,.?!\'" ])+' , ' ' , sent)
            tmp_vocab = tmp_vocab.split(' ')
            tmp_vocab = [ a for a in tmp_vocab if a!=' ' ]
            new_vocab.extend( tmp_vocab )
        
        new_vocab_set = set( new_vocab )
        new_vocab_list = [ a for a in new_vocab_set if new_vocab.count(a)>=1 ]
        random.shuffle( new_vocab_list )
        id_2_vocab.extend( new_vocab_list )

        for i in range( 0 , len(new_vocab_list) ):
            vocab_dict[ new_vocab_list[i] ] = vocab_index
            vocab_index += 1
        
        return vocab_dict , id_2_vocab

    def sent_to_id( self , sent , vocab_dict ):
        # sent_id = [ 2 ]
        sent_id = []
        for key in sent :
            if key in vocab_dict:
                sent_id.append( vocab_dict[key] )
            else :
                sent_id.append( 1 )
        # sent_id.append(3)
        return sent_id

    def make_cn_vocab_dict( self , sent_list ):
        id_2_vocab = [ 'PAD' , 'ST1' , 'EN1' ]
        vocab_dict = { 'PAD' : 0 , 'ST1' : 1 , 'EN1' : 2 }
        vocab_index = 3
        new_vocab = [ ]
        for sent in sent_list :
            sent = re.sub('[，。“”？]' , '' , sent)
            sent = cht_to_chs( sent )
            # tmp_vocab = jieba.cut(sent)
            # print( tmp_vocab )
            tmp_vocab = [ sent[j] for j in range( 0 , len(sent)) ]
            tmp_vocab = [ vocab for vocab in tmp_vocab if vocab!=' ' ]
            new_vocab.extend( tmp_vocab )
        
        new_vocab_set = set( new_vocab )pytho
        new_vocab_list = [ a for a in new_vocab_set if new_vocab.count(a)>=1 ]
        random.shuffle( new_vocab_list )
        
        id_2_vocab.extend( new_vocab_list )
        for i in range( 0 , len(new_vocab_list) ):
            vocab_dict[ new_vocab_list[i] ] = vocab_index
            vocab_index += 1
        
        return vocab_dict , id_2_vocab
        # 低频词当作UNK


    def read_data( self , filename='./dataset/cmn.txt' , re_write_map = False ):
        with open( './dataset/cmn.txt' , 'r' , encoding='utf-8' ) as f:
            lines = f.readlines()
        lines = lines[:10000]
        english = []
        chinese = []

        for line in lines:
            line = line.strip()
            line = line.split('\t')
            english.append( line[0] )
            chinese.append( line[1] )
        
        if re_write_map == True :
            print( 're_write' )
            eng_vocab_dict , eng_id_2_vocab = self.make_eng_vocab_dict( english )
            cn_vocab_dict , cn_id_2_vocab = self.make_cn_vocab_dict( chinese )

            with open( './dataset/eng_vocab_dict.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( eng_vocab_dict , f , ensure_ascii=False , indent=1 )
            with open( './dataset/eng_id_2_vocab.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( eng_id_2_vocab , f , ensure_ascii=False , indent=1 )
            with open( './dataset/cn_vocab_dict.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( cn_vocab_dict , f , ensure_ascii=False , indent=1 )
            with open( './dataset/cn_id_2_vocab.json' , 'w' , encoding='utf-8' ) as f:
                json.dump( cn_id_2_vocab , f , ensure_ascii=False , indent=1 )
        
        else :
            print( 'not_re_write' )
            with open( './dataset/eng_vocab_dict.json' , 'r' , encoding='utf-8' ) as f:
                eng_vocab_dict = json.load(f)
            with open( './dataset/eng_id_2_vocab.json' , 'r' , encoding='utf-8' ) as f:
                eng_id_2_vocab = json.load(f)
            with open( './dataset/cn_vocab_dict.json' , 'r' , encoding='utf-8' ) as f:
                cn_vocab_dict = json.load(f)
            with open( './dataset/cn_id_2_vocab.json' , 'r' , encoding='utf-8' ) as f:
                cn_id_2_vocab = json.load(f)
        
        chinese_sent_id = []
        english_sent_id = []
        for line in chinese:
            line = re.sub('[，。“”？]' , '' , line)
            line = cht_to_chs( line )
            # line = jieba.cut( line )
            line = [ line[j] for j in range( 0 , len(line)) ]
            chinese_sent_id.append( self.sent_to_id( line , cn_vocab_dict ) )
        
        for line in english :
            line = line.lower()
            line = re.sub('([,.?!\'"])+' , ' ' , line )
            line = line.split(' ')
            line = [ key for key in line if key!='' ]
            eng_sent_id = [1] + self.sent_to_id( line , eng_vocab_dict ) + [2]
            english_sent_id.append( eng_sent_id )
        
        # tmp_data = [ (chinese_sent_id[i] , english_sent_id[i]) for i in range( 0 , len(chinese_sent_id)) ]
        # random.shuffle( tmp_data )
        # chinese_sent_id = [ tmp_data[i][0] for i in range( 0 , len(tmp_data)) ]
        # english_sent_id = [ tmp_data[i][1] for i in range( 0 , len(tmp_data)) ]

        return chinese_sent_id , english_sent_id , cn_id_2_vocab , eng_id_2_vocab

