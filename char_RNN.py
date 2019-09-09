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


class Character_RNN():
    """
    This defines a character RNN for text generation
    """
    def __init__(self, **kwargs):
        self.rnn_size = kwargs['dim']
        self.learn_rate = kwargs['lr']
        self.maxlen = kwargs['maxlen']
        self.step = kwargs['step']
        self.epochs = kwargs['epochs']
        self.dropoutrate = kwargs['dropout']
        self.batch_size = kwargs['batchsize']
    
    def __repr__(self):
        print ("This is the Character RNN using BiLSTMs")
    

    def model_build(self, vocabulary_size):
        """Building the model
        """
        print ("Building the model ...")
        print ("Uses Bidirectional LSTM model without embedding layer ...")
        model = Sequential()
        model.add(Bidirectional(LSTM(self.rnn_size, activation="relu"), input_shape=(self.maxlen, vocabulary_size)))
        model.add(Dropout(self.dropoutrate))
        model.add(Dense(vocabulary_size))
        model.add(Activation("softmax"))
        optimizer = Adam(lr=self.learn_rate)
        # callbacks = [EarlyStopping(patience=2, monitor="val_loss")]
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[categorical_accuracy])
        model.summary(70) #print model summary
        return model


    def sample(self):
        """Helper function to sample an index from a probability array
        """
    
    def on_epoch_end(self, epoch, _):
        """Function invoked at end of each epoch. Prints generated text
        """
    
