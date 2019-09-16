import os
import sys
import textstat
import nltk

"""
Code to parse the documents/text in a given dataset. Evaluate the quality of the dataset.
--- readability scores
--- syllables per sentence/ syllables per word
--- characters "  "/ chars " "
--- 
"""

class EvalDQ():
    """
    """
    def __init__(self, filepath):
        self.path_ = filepath #the full file path to the dataset
        files_list = os.listdir(filepath) # 
        print (len(files_list))
        file_ = files_list[0]
        fullfilepath = os.path.join(filepath, file_)
        with open(fullfilepath, "r") as i_f:
            text_ = i_f.read()
            print (text_)
        # for file in filepath:
        #     with open(file, "r") as i_f:
        #         self.text_ =  #document as a list of sentences, tokenized into words
    
    def read_sent_stats(self):
        ##
        pass

