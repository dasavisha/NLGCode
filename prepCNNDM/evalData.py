import os
import sys
import textstat
import nltk
import spacy


nlp = spacy.load("en_core_web_lg")


"""
Code to parse the documents/text in a given dataset. Evaluate the quality of the dataset.
--- readability scores
--- syllables per sentence/ syllables per word
--- characters "  "/ chars " "
--- 
"""


class EvalDQ(object):
    """
    """
    def __init__(self, filepath):
        self.path_ = filepath #the full file path to the dataset
        
    def read_data_strings(self):
        ## read the entire document text as a string. 
        ## returns all the document data as a list
        files_list = os.listdir(self.path_) # 
        # print (len(files_list))
        file_data = list()
        for file_ in files_list:
            fullfilepath = os.path.join(self.path_, file_)
            with open(fullfilepath, "r") as i_f:
                text_ = i_f.read()
                file_data.append(text_)
                # print (text_)
        return file_data, len(files_list)

    def sentence_tokenizer(self):
        ##returns a list of lists
        ## each nested list is a list of sentence tokens in a document
        file_data, _ = self.read_data_strings() #get the text documents as a list
        file_data_sents = list()
        for data in file_data:
            data_sents = list()
            doc = nlp(data)
            for token in doc.sents:
                data_sents.append(token.text)
            file_data_sents.append(data_sents)
        return file_data_sents
    
    def word_tokenizer(self):
        ## returns a list of lists
        ## each nested list is a list of word tokens in a document
        file_data, _ = self.read_data_strings() #get the text documents as a list
        file_data_words = list()
        file_data_tokens = list()
        for data in file_data:
            data_words = list()
            data_tokens = list()
            doc = nlp(data)
            for token in doc:
                data_words.append(token.text)
                data_tokens.append(token) 
            file_data_words.append(data_words)
            file_data_tokens.append(data_tokens)
        return file_data_words, file_data_tokens
    
    def count_contentwords(self):
        ## count the total of content words in the data
        ## also returns the total of stopwords 
        _, file_data_tokens = self.word_tokenizer()
        stopword_ct_list = list()
        contentword_ct_list = list()
        for data_list in file_data_tokens:
            total_tokens = len(data_list) #number of word tokens in each list
            stopword_ctr = 0
            for token in data_list:
                if token.is_stop:
                    stopword_ctr += 1
            contentword_ctr = total_tokens - stopword_ctr
            contentword_ct_list.append(contentword_ctr)
            stopword_ct_list.append(stopword_ctr)
        return contentword_ct_list, stopword_ct_list
    
    def get_readability_scores(self):
        ## read the list of sentences in each document 
        ## calculate and return the dictionary of scores
        file_data_sents = self.sentence_tokenizer()
        file_rdscores_list = list()
        for sent_lists in file_data_sents:
            rdscores_dict = {}
            rdscores_dict['fl_rdng_ease'] = textstat.flesch_reading_ease(sent_lists)
            rdscores_dict['smog_idx'] = textstat.smog_index(sent_lists)
            rdscores_dict['fl_kincaid_grd'] = textstat.flesch_kincaid_grade(sent_lists)
            rdscores_dict['coleman_liau_idx'] = textstat.coleman_liau_index(sent_lists)
            rdscores_dict['auto_rd_idx'] = textstat.automated_readability_index(sent_lists)
            rdscores_dict['dc_rd_score'] = textstat.dale_chall_readability_score(sent_lists)
            # difficult_words = textstat.difficult_words(test_data)
            rdscores_dict['lw_formula'] = textstat.linsear_write_formula(sent_lists)
            rdscores_dict['gunning_fog'] = textstat.gunning_fog(sent_lists)
            # = textstat.text_standard(test_data)
            file_rdscores_list.append(rdscores_dict)
        return file_rdscores_list

    def get_syllable_scores_sent(self):
        ##
        ##
        file_data_sents = self.sentence_tokenizer()
        file_sylsents_list = list()
        for sent_lists in file_data_sents:
            val = 0
            for sent in sent_lists:
                val += textstat.syllable_count(sent, lang='en_US')
            final_val = float(val)/len(sent_lists)
            file_sylsents_list.append(final_val)
        return file_sylsents_list
    
    def get_syllable_scores_word(self):
        ##
        ##
        file_data_words = self.word_tokenizer()
        file_sylwords_list = list()
        for word_lists in file_data_words:
            val = 0
            for word in word_lists:
                val += textstat.syllable_count(word, lang='en_US')
            final_val = float(val)/len(word_lists)
            file_sylwords_list.append(final_val)
        return file_sylwords_list
                
    def get_characters_sent(self):
        ##
        ##
        pass

    def get_characters_word(self):
        ##
        ##
        pass

    def word2vec_sentsim(self):
        ##
        ##
        pass

    def wordnet_sim(self):
        ##
        ##
        pass

    def ngram_overlap(self, n=3):
        ##
        ##
        pass
    