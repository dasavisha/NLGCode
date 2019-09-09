import os
import sys
import math
import random as rd
from keras.models import Sequential, Model 
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras_data_batchgenerator import DataGenerator

import spacy
nlp = spacy.load('en_core_web_lg') #for text tokenization

import numpy as np
import random
import time
import codecs
import collections
import pickle
import joblib
from tempfile import TemporaryFile

tokenized_words = list()

save_dir = '../save/'
word_tokens_file = os.path.join(save_dir, 'vocabulary_keras.pkl') #saving the tokenized words
vocabulary_file = os.path.join(save_dir, 'wordtokens_keras.pkl') #saving the vocabulary


def create_wordlist(in_text):
    	#tokenization of text chapter
    wordList = []
    for word in in_text:
        if word.text not in ("\n","\n\n",'\u2009','\xa0'):
            wordList.append(word.text.lower())
    return wordList


def data_prep_words(self, path, typeFile=False, typeDir=False):
    """Data preprocessing step for vectorizing the data
    """
    #read the file
    print("Reading the data ...")
    if typeFile:
        data_input_file = path
        with codecs.open(data_input_file, "r") as i_f:
            # data_input = i_f.read()
            counter = 0
            for data_input in i_f: #read eachline
                input_text = nlp(data_input)
                counter += 1
                if counter % 10000 == 0:
                    print ("Done, {}".format(counter))
                wl = create_wordlist(input_text) #get word list of entire chapter 
                tokenized_words.extend(wl) # insert the word list 

    if typeDir:
        list_files = os.listdir(path)
        for file in list_files:
            data_input_file = os.path.join(path, file)
            with codecs.open(data_input_file, "r") as i_f:
                data_input = i_f.read()
            input_text = nlp(data_input)
            wl = create_wordlist(input_text) #get word list of entire chapter 
            tokenized_words.extend(wl) # insert the word list 
    #write to a tokenized_words file 
    with open(word_tokens_file, 'wb') as f_wt: #save word tokens
        pickle.dump(tokenized_words, f_wt)
    return (tokenized_words)

def word_indexing():
    print ("Word Tokens: ", len(tokenized_words))
    ###create a word indexing system
    word_counts = collections.Counter(tokenized_words)
    #mapping index to word - for building a dictionary
    ##gets the list of words based on their frequency of occurence
    words_ = [x[0] for x in word_counts.most_common()] #get the words arranged in descending order of counts
    vocab_inv = list(sorted(words_)) #get the words in a list

    ##mapping from word to index -- 
    vocabulary_ = {x: i for i,x in enumerate(vocab_inv)} #get the word to index using list indexing

    # size of the vocabulary
    vocabulary_size = len(words_) #len(vocabulary_) #
    print ("Vocabulary size: ", vocabulary_size)
    #write to a vocabulary file
    with open(vocabulary_file, 'wb') as f: #save vocabulary and words
        pickle.dump((words_, vocabulary_, vocab_inv), f)
