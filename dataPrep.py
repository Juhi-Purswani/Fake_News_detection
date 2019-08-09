# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


from matplotlib import pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
import re

data_train=pd.read_csv('data/train2.tsv',delimiter='\t',encoding='utf-8',names=["id", "label", "title", "Coverage","i", "lal", "Ed", "Cerage","d", "lbel", "Egtd", "Covage","Dvds","cds","text"])
data_validate=pd.read_csv('data/val2.tsv',delimiter='\t',encoding='utf-8',names=["id", "label", "title", "Coverage","i", "lal", "Ed", "Cerage","d", "lbel", "Egtd", "Covage","Dvds","cds","text"])

data_train.drop(['id','Coverage','i','lal','Ed','Cerage','d','lbel','Egtd','Covage','Dvds','cds'],axis=1,inplace=True)
data_validate.drop(['id','Coverage','i','lal','Ed','Cerage','d','lbel','Egtd','Covage','Dvds','cds'],axis=1,inplace=True)


data_train.title = data_train.title.str.lower()
data_train.text = data_train.text.str.lower()
data_train.title = data_train.title.str.replace(r'\s\s+',' ')
data_train.text = data_train.text.str.replace(r'\s\s+',' ')
data_train.title = data_train.title.str.replace(r'[^\.\w\s]','') #remove everything but characters and punctuation
data_train.text = data_train.text.str.replace(r'[^\.\w\s]','')
data_train.title = data_train.title.str.replace(r'\.\.+','.') #replace multple periods with a single one
data_train.text = data_train.text.str.replace(r'\.\.+','.')
data_train.title = data_train.title.str.strip() 
data_train.text = data_train.text.str.strip()
#data_train.title = re.sub(r '\d+' , '' , data_train.title)
#data_train.text = re.sub(r '\d+' , '' , data_train.text)
#data_train.title = ''.join([i for i in data_train.title if not i.isdigit()])

data_train.title = data_train.title.str.replace('\d+', '')
data_train.text = data_train.text.str.replace('\d+', '')


data_validate.title = data_validate.title.str.lower()
data_validate.text = data_validate.text.str.lower()
data_validate.title = data_validate.title.str.replace(r'\s\s+',' ')
data_validate.text = data_validate.text.str.replace(r'\s\s+',' ')
data_validate.title = data_validate.title.str.replace(r'[^\.\w\s]','') #remove everything not characters and punctuation
data_validate.text = data_validate.text.str.replace(r'[^\.\w\s]','')
data_validate.title = data_validate.title.str.replace(r'\.\.+','.') #multple periods with a single one
data_validate.text = data_validate.text.str.replace(r'\.\.+','.')
data_validate.title = data_validate.title.str.strip() 
data_validate.text = data_validate.text.str.strip()
#data_train.title = re.sub(r '\d+' , '' , data_train.title)
#data_train.text = re.sub(r '\d+' , '' , data_train.text)
#data_train.title = ''.join([i for i in data_train.title if not i.isdigit()])

data_validate.title = data_validate.title.str.replace('\d+', '')
data_validate.text = data_validate.text.str.replace('\d+', '')

data_train

print(data_train.isnull().sum())
print(data_validate.isnull().sum())

data_train.dropna()
data_validate.dropna()

texts_train = []
labels_train = []

for i in range(data_train.text.shape[0]):
    text1 = data_train.title[i]
    text2 = data_train.text[i]
    text = str(text1) +""+ str(text2)
    texts_train.append(text)
    labels_train.append(data_train.label[i])
    
for i in range(len(labels_train)):
    if(labels_train[i]=='mostly-true'):
        labels_train[i] = 'true'
    
    elif(labels_train[i]=='barely-true'):
        labels_train[i] = 'false'
    
    elif(labels_train[i]=='pants-fire'):
        labels_train[i] = 'false'
    elif(labels_train[i]=='half-true'):
        labels_train[i] = 'true'
        
        
texts_validate = []
labels_validate = []

for i in range(data_validate.text.shape[0]):
    text1 = data_validate.title[i]
    text2 = data_validate.text[i]
    text = str(text1) +""+ str(text2)
    texts_validate.append(text)
    labels_validate.append(data_validate.label[i])
    
for i in range(len(labels_validate)):
    if(labels_validate[i]=='mostly-true'):
        labels_validate[i] = 'true'
    
    elif(labels_validate[i]=='barely-true'):
        labels_validate[i] = 'false'
    
    elif(labels_validate[i]=='pants-fire'):
        labels_validate[i] = 'false'
    elif(labels_validate[i]=='half-true'):
        labels_validate[i] = 'true'

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)

word_index = tokenizer.word_index
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

df = pd.DataFrame(labels_train) 
y_train = pd.get_dummies(df)

print('Shape of data tensor:', x_train)
print('Shape of label tensor:', y_train)

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

#for validation data
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_validate)
sequences = tokenizer.texts_to_sequences(texts_validate)

#word_index = tokenizer.word_index
x_validate = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

df = pd.DataFrame(labels_validate) 
y_validate = pd.get_dummies(df)

print('Shape of data tensor:', x_validate)
print('Shape of label tensor:', y_validate)

indices = np.arange(x_validate.shape[0])
np.random.shuffle(indices)
x_validate = x_validate[indices]
y_validate = y_validate[indices]

def data():
    return(x_train,y_train,x_validate,y_validate)