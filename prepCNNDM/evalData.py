import os
import sys
from textstat.textstat import textstat
import nltk
from nltk.util import ngrams
import spacy
import re
import pandas as pd
import statistics
import gensim
from gensim.models import KeyedVectors
import numpy as np
from scipy import spatial
# from nltk.parse.corenlp import CoreNLPServer
# from nltk.parse.corenlp import CoreNLPDependencyParser
from bllipparser import RerankingParser
from scipy.stats import kurtosis, skew
import collections        
from nltk.corpus import stopwords
from collections import Counter

# Get the list of english stopwords from English
stop_words = set(stopwords.words('english'))

# # Load Google's pre-trained Word2Vec model.
# w2v_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
# index2word_set = set(w2v_model.index2word)

# Load Spacy's language model
nlp = spacy.load("en_core_web_lg")

# # # Create the CoreNLP server for dependency parsing
# # STANFORD = os.path.join("../stories_data", "stanford-corenlp-full-2018-02-27")
# # server = CoreNLPServer(
# #    os.path.join(STANFORD, "stanford-corenlp-3.9.1.jar"),
# #    os.path.join(STANFORD, "stanford-corenlp-3.9.1-models.jar"),    
# # )

#downloads, installs, and loads the model for Bllipparsing
rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2', verbose=True)

"""
Code to parse the documents/text in a given dataset. Evaluate the quality of the dataset.
--- readability scores
--- syllables per sentence/ syllables per word
--- characters "  "/ chars " "
--- 
"""


class EvalDQ(object):
    """
    Evaluation metrics used in the paper have been implemented here
    """
    def __init__(self, filepath):
        self.path_ = filepath #the full file path to the dataset
        
    def read_data_strings(self):
        temp_regex = r"T[0-9]+"
        topk_regex = r"k[0-9]+"
        idx_regex = r"_[0-9]+"
        tag_prp = "Prompt: "
        tag_ori = "Original: "
        tag_gen = "Generated: "
        tag_time = "Time elapsed: "
        dfObj = pd.DataFrame(columns=['idx', 'temp', 'topK', 'prompt', 'original', 'generated'])
        def read_btw_twotags(s, tag1, tag2):
            """code to read text between two tags
                tag1-starting tag; tag2-ending tag
                #https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
            """
            try:
                start = s.find(tag1) + len(tag2)
                end = s.rfind(tag2)
                return s[start:end]
            except ValueError:
                return ""
        
        ## read the entire document text as a string.
        ## separate out the "prompt", "original" and the "generated" 
        ## returns all the document data as a list of tuples
        files_list_permodel = os.listdir(self.path_) #path to the sample files for every model
        # print (len(files_list))
        # file_data = list()
        for file_ in files_list_permodel:
            fullfilepath = os.path.join(self.path_, file_)
            temp = re.findall(temp_regex,file_) #get temperature from filename
            topk = re.findall(topk_regex,file_) #get temperature from filename
            idx_ = re.findall(idx_regex, file_) #get the index as a string "_\d"
            with open(fullfilepath, "r") as i_f:
                text_ = i_f.read() #read the text as a string
                prompt_text = read_btw_twotags(text_, tag_prp, tag_ori) #get the prompt
                original_text = read_btw_twotags(text_, tag_ori, tag_gen) #get the original story
                generated_text = read_btw_twotags(text_, tag_gen, tag_time) #get the generated story
                dfObj = dfObj.append({'idx': (idx_[0]).replace('_',''), 'temp': temp[0], 'topK': topk[0], 'prompt': prompt_text, 'original': original_text, 'generated': generated_text}, ignore_index=True)
                # data_tuple = ()
                # file_data.append(text_) 
                # print (text_)
        return dfObj

    def read_fusion_output(self):
        id_regex = r"-\d+" #regex for extracting the source ids
        tag_prp = "S-"
        tag_ori = "T-"
        tag_gen = "H-"
        dfObj_fusion = pd.DataFrame(columns=['idx', 'temp', 'topK', 'prompt', 'original', 'generated'])
        filepath_prompts = os.path.join(self.path_, "fusion_stories_self_prompts.txt") 
        filepath_original = os.path.join(self.path_, "fusion_stories_self_original.txt")
        filepath_generated = os.path.join(self.path_, "fusion_stories_self_generated.txt")
        #get the ids present in the prompts file
        src_ids_fus = list()  
        prompts_fus = list()
        with open(filepath_prompts, "r") as s_f:
            text_ = s_f.readline()
            prompt = text_.split(' ', 1)[1]
            src_id = re.findall(id_regex, text_)
            src_id = (src_id[0]).replace('-', '')
            src_ids_fus.append(src_id)
            prompts_fus.append(prompt)
        print (len(src_ids_fus))
        
        with open(filepath_original, "r") as o_f:
            text_ori = o_f.readlines()
        text_ori = [x.strip() for x in text_ori]
        original_fus = list()
        for idx in src_ids_fus:
            tag_req_o = tag_ori+idx 
            for ln in text_ori:
                if ln.startswith(tag_req_o):
                    ori =  ln.split(' ', 1)[1]
                    original_fus.append(ori)
                else:
                    continue

        with open(filepath_generated, "r") as g_f:
            text_gen = g_f.readlines()
        text_gen = [x.strip() for x in text_gen]
        generated_fus = list()
        for idx in src_ids_fus:
            tag_req_g = tag_gen+idx 
            for ln in text_gen:
                if ln.startswith(tag_req_g):
                    gen =  ln.split(' ', 1)[1]
                    generated_fus.append(gen)
                else:
                    continue
        
        dfObj_fusion["idx"] = src_ids_fus
        temp = pd.Series([1.0]*len(src_ids_fus))
        dfObj_fusion["temp"] = temp.values
        topk = pd.Series([10]*len(src_ids_fus))
        dfObj_fusion["topK"] = topk.values
        dfObj_fusion["original"] = original_fus
        dfObj_fusion["generated"] = generated_fus

        return dfObj_fusion

    def sentence_tokenizer(self, file_data):
        ##returns a list of lists
        ## each nested list is a list of sentence tokens in a document
        ## get the text documents as a list
        file_data_sents = list()
        for data in file_data:
            data_sents = list()
            doc = nlp(data)
            for token in doc.sents:
                data_sents.append(token.text)
            file_data_sents.append(data_sents)
        return file_data_sents
    
    def word_tokenizer(self, file_data, sw=False, stem=False):
        ## returns a list of lists
        ## each nested list is a list of word tokens in a document
        file_data_words = list()
        # file_data_tokens = list()
        for data in file_data:
            data_words = list()
            data = data.lower()
            # data_tokens = list()
            doc = nlp(data)
            for token in doc:
                data_words.append(token.text)
                # data_tokens.append(token) 
            file_data_words.append(data_words)
            # file_data_tokens.append(data_tokens)
        if sw: #if stopword elimination is set 
            # pass
            print ("Doing stopword elimination ...")
            file_data_words_sw = list()
            for wordlist in file_data_words:
                wordlist_sw = list()
                for token in wordlist:
                    if token not in stop_words:
                        wordlist_sw.append(token)
                file_data_words_sw.append(wordlist_sw)
            file_data_words = file_data_words_sw
        if stem: #if stemming is set
            pass
        return file_data_words
    
    def get_readability_scores(self, dfObj, type_text):
        ## read the data from the df object passed 
        ## calculate and return the dictionary of scores
        ## this is a measure of the grammaticality score of the text
        if type_text == "original":
            file_data = dfObj["original"].tolist()
        if type_text == "generated":
            file_data = dfObj["generated"].tolist()

        file_data_sents = self.sentence_tokenizer(file_data)
        file_rdscores_list = list()
        for sent_lists in file_data_sents:
            rdscores_dict = {}
            # print (sent_lists)
            sent_str = ' '.join(sent_lists)
            try:
            # fl_rdng_ease = textstat.flesch_reading_ease(sent_str)
            # print (fl_rdng_ease)
                rdscores_dict['fl_rdng_ease'] = textstat.flesch_reading_ease(sent_str) #throwing errors in transformer outputs
                # exit()
                rdscores_dict['smog_idx'] = textstat.smog_index(sent_str)
                rdscores_dict['fl_kincaid_grd'] = textstat.flesch_kincaid_grade(sent_str)
                rdscores_dict['coleman_liau_idx'] = textstat.coleman_liau_index(sent_str)
                rdscores_dict['auto_rd_idx'] = textstat.automated_readability_index(sent_str)
                rdscores_dict['dc_rd_score'] = textstat.dale_chall_readability_score(sent_str)
                # difficult_words = textstat.difficult_words(test_data)
                rdscores_dict['lw_formula'] = textstat.linsear_write_formula(sent_str)
                rdscores_dict['gunning_fog'] = textstat.gunning_fog(sent_str)
                # = textstat.text_standard(test_data)
                file_rdscores_list.append(rdscores_dict)
            except TypeError:
                continue
        dfObj_RS = pd.DataFrame(file_rdscores_list)
        return dfObj_RS ####read to df

    def count_contentwords(self, dfObj, type_text):
        ## count the normalized value of content words in the data
        ## also returns the normalized value of stopwords 
        file_data = dfObj[type_text].tolist()
        stopword_ct_list = list()
        contentword_ct_list = list()
        file_data_words = self.word_tokenizer(file_data)
        for token_list in file_data_words:
            total_tokens = len(token_list) #number of word tokens in each list
            stopword_ctr = 0
            contentword_ctr = 0
            for token in token_list:
                if token in stop_words:
                    stopword_ctr += 1
                else:
                    contentword_ctr += 1
            # print (contentword_ctr, stopword_ctr)
            # exit()
            try:
                sw_ctr_norm = float(stopword_ctr)/total_tokens
                cw_ctr_norm = float(contentword_ctr)/total_tokens
                contentword_ct_list.append(cw_ctr_norm)
                stopword_ct_list.append(sw_ctr_norm)
            except ZeroDivisionError:
                continue
        # print (contentword_ct_list)
        return contentword_ct_list, stopword_ct_list
    
    def ngram_overlap(self, dfObj, type_tuple, n=3, sw=True, stem=True):
        ## percentage of n-gram overlap between two given strings 
        ## the type_tuple gives the two string types to look at for calculating 
        ## this is a measure of the story-prompt relatedness 
        ## note: pc is calculated wrt to the first type given
        file_data_t1 = dfObj[type_tuple[0]].tolist()
        file_data_t2 = dfObj[type_tuple[1]].tolist() 
        
        file_data_t1_words = self.word_tokenizer(file_data_t1, sw=sw, stem=stem)
        file_data_t2_words = self.word_tokenizer(file_data_t2, sw=sw, stem=stem)
        overlap_pc_list = list()
        print ("Calculating the overlap between the grams from {} and {} for n={} ...".format(type_tuple[0], type_tuple[1], n))
        for wl_t1, wl_t2 in zip(file_data_t1_words, file_data_t2_words):
            ngrams_t1 = list(ngrams(wl_t1, n))
            ngrams_t2 = list(ngrams(wl_t2, n))
            common_ngrams = [value for value in ngrams_t1 if value in ngrams_t2]
            try:
                overlap_pc = (float(len(common_ngrams))/len(ngrams_t1))*100
                overlap_pc_list.append(overlap_pc)
            except ZeroDivisionError:
                continue
        return (statistics.mean(overlap_pc_list))

    def word2vec_sentsim(self, dfObj, type_tuple):
        ## cosine similarity between the story and prompt word vectors averaged over all pairs
        ## get the word2vec representation of the story and prompts
        ## this is a measure of the story-prompt relatedness
        ## https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt
        def avg_sentence_vector(words, model, num_features, index2word_set):
            #function to average all words vectors in a given paragraph
            featureVec = np.zeros((num_features,), dtype="float32")
            nwords = 0
            for word in words:
                if word in index2word_set:
                    nwords = nwords+1
                    featureVec = np.add(featureVec, model[word])
            if nwords>0:
                featureVec = np.divide(featureVec, nwords)
            return featureVec
        
        file_data_t1 = dfObj[type_tuple[0]].tolist()
        file_data_t2 = dfObj[type_tuple[1]].tolist() 
        
        file_data_t1_sents = self.sentence_tokenizer(file_data_t1)
        file_data_t2_sents = self.sentence_tokenizer(file_data_t2)

        file_data_t1_meansim = list()
        for sent_lists_t1, sent_lists_t2 in zip(file_data_t1_sents, file_data_t2_sents): 
            prp_str = ' '.join(sent_lists_t2)
            sent_t2_avg_vector = avg_sentence_vector(prp_str.split(), model=w2v_model, num_features=100, index2word_set=index2word_set)
            perdoc_simlist = list()
            for sent in sent_lists_t1:
                sent_t1_avg_vector = avg_sentence_vector(sent.split(), model=w2v_model, num_features=100, index2word_set=index2word_set)
                sim = 1 - spatial.distance.cosine(sent_t1_avg_vector, sent_t2_avg_vector)
                perdoc_simlist.append(sim)
            mean_sim = statistics.mean(perdoc_simlist)
            file_data_t1_meansim.append(mean_sim)
        
        return (statistics.mean(file_data_t1_meansim))
        
    def average_sentence_length(self, dfObj, type_text):
        ## returns the length of each story given a model
        ## as a list
        ## a measure of the syntactic style and complexity
        if type_text == "original":
            file_data = dfObj["original"].tolist()
        if type_text == "generated":
            file_data = dfObj["generated"].tolist()
        file_data_sents = self.sentence_tokenizer(file_data)
        file_sentlen_list = list()
        for sent_lists in file_data_sents:
            ctr = 0
            for i in sent_lists:
                sl_i = len(i.split()) #length of each sentence in the text
                ctr+=sl_i #update the counter
            try:
                avg_sl = float(ctr)/len(sent_lists) #calculate the average sentence length by dividing by the total number of sentences
                file_sentlen_list.append(avg_sl) 
            except ZeroDivisionError:
                continue
        return file_sentlen_list #return the list of average sentence lengths of each story
            

    def parsing_score_calculation(self, dfObj, type_text):
        ## This uses the constituency parsing score returned by the BLLIP parser
        ## We use the bllipparser to get the negative log probabilities 
        ## We report the skewness and kurtosis of the 50 top parses
        ## We return three lists of mean parse scores across the documents, 
        ## the skewness of the parse scores and the kurtosis of the parse scores
        file_data = dfObj[type_text].tolist()
        file_data_sents = self.sentence_tokenizer(file_data)
        mean_p_scores_alldocs = list() #parse scores stored as a list for all the documents
        skew_p_scores_alldocs = list() #skewness of parse scores
        kurt_p_scores_alldocs = list() #kurtosis of parse scores
        for sent_lists in file_data_sents:
            for sent_i in sent_lists:
                p_scorelist_perdoc = list()
                nbest_list = rrp.parse(sent_i)
                for p_id in range(len(nbest_list)): 
                    try:
                        p_score = nbest_list[p_id].parser_score
                        p_scorelist_perdoc.append(p_score)
                    except statistics.StatisticsError:
                        continue
                    mean_p_score = statistics.mean(p_scorelist_perdoc) #the mean of all the 50 parse scores
                    skewness_pscores = skew(p_scorelist_perdoc) #the skewness of the parse scores
                    kurtosis_pscores = kurtosis(p_scorelist_perdoc) #the kurtosis of the parse scores
            mean_p_scores_alldocs.append(mean_p_score)
            skew_p_scores_alldocs.append(skewness_pscores)
            kurt_p_scores_alldocs.append(kurtosis_pscores)
        return mean_p_scores_alldocs, skew_p_scores_alldocs, kurt_p_scores_alldocs    

    
    def get_rareword_usage(self, dfObj):
        ## get the unigram probability of the words in the generated text 
        ## calculate the mean log of the probability
        original_data = dfObj["original"].tolist()
        tokens = list()
        for story in original_data:
            tokens.extend(nltk.word_tokenize(story)) #add to the list
        # building the unigram probability model:
        # https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
        model = collections.defaultdict(lambda: 0.01)
        for f in tokens:
            try:
                model[f] += 1
            except KeyError:
                model[f] = 1
                continue
        N = float(sum(model.values()))
        for word in model:
            model[word] = model[word]/N
        # extract the mean log probability of the generated text
        generated_file_data = dfObj["generated"].tolist()
        file_data_words = self.word_tokenizer(generated_file_data)
        mean_ug_prb_file = list()
        for word_list in file_data_words:
            ug_prb_perstory = list()
            for w in word_list:
                ug_prb = model[w] 
                ug_prb_perstory.append(ug_prb)
            mean_ug_prb_perstory = statistics.mean(ug_prb_perstory)
            mean_ug_prb_file.append(mean_ug_prb_perstory)
        return mean_ug_prb_file
    
    def pos_tag_freqdist(self, dfObj, type_text):
        ## determine the percentages of the various pos tags
        ## https://www.guru99.com/counting-pos-tags-nltk.html
        file_data = dfObj[type_text].tolist()
        pos_tokens_pc_file = list() #list of POS tags percentages dictionary
        for text in file_data:
            word_tokens_pos = list()
            doc = nlp(text)
            for token in doc:
                tup = (token.text, token.pos_)
                word_tokens_pos.append(tup)    
            # word_tokens = nltk.word_tokenize(text)
            # word_tokens_pos = nltk.pos_tag(word_tokens)
            pos_counts = Counter(tag for word, tag in word_tokens_pos) #dictionary of p-o-s counts 
            pos_pc = {k : v / float(len(word_tokens_pos)) for k,v in pos_counts.items()}
            pos_tokens_pc_file.append(pos_pc) 
        ## based on the required tag get the count of the tags
        ## convert the list of dictionaries of POS tag percentages into DF
        ## calculate the average of each column
        dfObj_POS = pd.DataFrame(pos_tokens_pc_file)
        return dfObj_POS
         

    # def count_misspellings(self, dfObj, type_text):
    #     pass 

    
    # def get_syllable_scores_sent(self):
    #     ##
    #     ##
    #     file_data_sents = self.sentence_tokenizer()
    #     file_sylsents_list = list()
    #     for sent_lists in file_data_sents:
    #         val = 0
    #         for sent in sent_lists:
    #             val += textstat.syllable_count(sent, lang='en_US')
    #         final_val = float(val)/len(sent_lists)
    #         file_sylsents_list.append(final_val)
    #     return file_sylsents_list
    
    # def get_syllable_scores_word(self):
    #     ##
    #     ##
    #     file_data_words = self.word_tokenizer()
    #     file_sylwords_list = list()
    #     for word_lists in file_data_words:
    #         val = 0
    #         for word in word_lists:
    #             val += textstat.syllable_count(word, lang='en_US')
    #         final_val = float(val)/len(word_lists)
    #         file_sylwords_list.append(final_val)
    #     return file_sylwords_list
                
    # def get_characters_sent(self):
    #     ##
    #     ##
    #     pass

    # def get_characters_word(self):
    #     ##
    #     ##
    #     pass


    # def wordnet_sim(self):
    #     ##
    #     ##
    #     pass

    


def main():
    sample_dirpath = sys.argv[1] #path to the samples
    model_name = "gpt2" #model name
    sample_modelpath = os.path.join(sample_dirpath, model_name)
    pass    



if __name__ == '__main__':
    main()