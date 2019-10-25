from evalData import EvalDQ
from file_preprocessor import FilePreprocessor
import os
import sys
import argparse
import logging
from tqdm import trange
from run_generation import text_generator
from nltk import word_tokenize
import time
import random 
import statistics


def truncate_string(text, length):
    """
    truncate the string to the given number of words
    """
    word_tokens = word_tokenize(text)
    truncated = word_tokens[:length]
    truncated_text = " ".join(truncated)
    return truncated_text

def main():
    """
    give the path to the directory with the documents
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default=None, type=str, required=True, help="Path to dataset")
    parser.add_argument("--truncate", action='store_true', help="Truncate the data when enabled")
    parser.add_argument("--stats", action='store_true', help="Get stats for the file")
    parser.add_argument("--count_vocab", action='store_true', help="Get vocabulary count and save vocabulary for the file")
    ##generation
    parser.add_argument('--generate', action='store_true', help="Start the generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature setting")
    parser.add_argument("--length", type=int, default=150, help="number of words to be generated")
    parser.add_argument("--top_k", type=int, default=1, help="parameter for Top-k sampling")
    parser.add_argument('--stop_token', type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples to be generated and compared with")
    parser.add_argument('--save_dir', default="../save/", type=str, help="Path to save the system outputs")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    ##evaluation
    parser.add_argument("--evaluate", action='store_true', help="Start the evaluation")
    parser.add_argument("--eval_dir", default='../save/gpt2/', help="The path to evaluate the system outputs")
    parser.add_argument("--eval_model", default='gpt2', help="The model name to evaluate the system outputs")
    parser.add_argument("--reading_scores", action='store_true', help="Get the average reading scores") #OK
    parser.add_argument("--content_words", action='store_true', help="Get the normalized mean of content words and stop words") #OK
    parser.add_argument("--ngram_overlap", action='store_true', help="Get the average N gram overlap percentage with the prompt") #OK
    parser.add_argument("--sw", action='store_true', help="Do stopword elimination")
    parser.add_argument("--stem", action='store_true', help="Do stemming")
    parser.add_argument("--parse_scores", action='store_true', help="Get the average, skewness and kurtosis of the parses of stories") 
    parser.add_argument("--sentemb_sim_scores", action='store_true', help="Get the sentence embedding similarity percentage with the prompt")
    parser.add_argument("--sent_length", action='store_true', help="Get the average sentence length")
    parser.add_argument("--pos_tag_fqd", action='store_true', help="Get POS tag frequency distribution as percentages")
    parser.add_argument("--log_unigm_prob", action='store_true', help="Get the average log unigram probability")
    # parser.add_argument("--coherence_scores", action='store_true', help="Get the average coherence scores") 
    args = parser.parse_args()


    filepath = args.filepath
    truncate_bool = args.truncate
    stats_bool = args.stats 
    vocab_bool = args.count_vocab
    #generation
    generate_bool = args.generate
    temperature = args.temperature
    length = args.length
    top_k = args.top_k
    stop_token = args.stop_token
    num_samples = args.num_samples
    save_dir = args.save_dir
    no_cuda_bool = args.no_cuda
    #evaluation
    evaluate_bool = args.evaluate
    eval_direcpath = args.eval_dir #path to the model folder
    eval_modelname = args.eval_model #name of the model evaluating
    eval_RS = args.reading_scores #evaluate reading scores
    eval_CW = args.content_words #evaluate the percentage of content and stop words
    eval_NG = args.ngram_overlap #evaluate story prompt relatedness scores with ngram overlap pc
    eval_PS = args.parse_scores #evaluate the grammaticality
    eval_SE = args.sentemb_sim_scores #evaluate story prompt relatedness scores
    eval_SL = args.sent_length #evaluate the syntactic complexity
    eval_PF = args.pos_tag_fqd #evaluate the pos-tag frequency distribution as percentages
    eval_RW = args.log_unigm_prob #evaluate the rareword usage scores as mean log unigram probability
    sw = False
    if args.sw:
        sw = True
    stem = False
    if args.stem:
        stem = True

    f_prep = FilePreprocessor(filepath) 
    if truncate_bool: #required when you are running the code the first time
        f_prep.truncate_stories(num_words=1000)
    if stats_bool:
        num_stories, num_prompts = f_prep.check_num_stories()
        print (num_prompts, num_stories)    
    if vocab_bool:
        vocab_counter_prompt, vocab_counter_story = f_prep.make_vocabulary()
        print ("The vocabulary for the stories: {}".format(vocab_counter_story))
        print ("The vocabulary for the prompts: {}".format(vocab_counter_prompt))
    ##### get the prompt from the file -- done
    ##### get the model type and model file name and path as a dictionary -- done
    ##### for each model type save the prompt, the original story and the generated story with "temp val" and "top k" val  and "model name" and "index of random story prompt selected" in a file: "gentext_"+model_+"_"+temperature+"_"+top_k+"_"+i  -- done
    ##### finish the 4 openai gptx models and then move onto xlnet models --done
    if generate_bool:
        # define the pre-trained models offered by huggingface/transformers github: https://github.com/huggingface/transformers for generation
        # Model classes at https://github.com/huggingface/transformers/blob/master/examples/run_generation.py 
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        # PT_model_dict = {"openai-gpt": ["openai-gpt"], "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"], "xlnet": ["xlnet-base-cased", "xlnet-large-cased"], "transfo-xl": ["transfo-xl-wt103"], "xlm": ["xlm-mlm-en-2048", "xlm-mlm-ende-1024", "xlm-mlm-enfr-1024", "xlm-mlm-enro-1024", "xlm-mlm-tlm-xnli15-1024", "xlm-mlm-xnli15-1024", "xlm-clm-enfr-1024", "xlm-clm-ende-1024", "xlm-mlm-17-1280", "xlm-mlm-100-1280"]}
        PT_model_dict = {"openai-gpt": ["openai-gpt"], "gpt2": ["gpt2", "gpt2-medium", "gpt2-large"], "xlnet": ["xlnet-base-cased", "xlnet-large-cased"], "transfo-xl": ["transfo-xl-wt103"]}
        # #check values for variables exist
        # assert temperature
        # assert length
        # assert top_k
        print ("Get the prompts from {} samples in the test set...".format(num_samples))
        story_files_dict = f_prep.get_art_prp_file()
        story_files_test = story_files_dict['test']
        nums_selected = random.sample(range(len(story_files_test)), num_samples)
        for idx, i in enumerate(nums_selected):
            prompt = (story_files_test[i][0]).replace("[ wp ]", "") #remove the tag from the prompt and save it
            story = story_files_test[i][1]
            # print ("Prompt: {}".format(prompt))
            # print ("Original Story: {}".format(story))
            for k,v in PT_model_dict.items():
                model_type = k
                model_names_list = v
                for model_ in model_names_list:
                    print ("Generating story #{} with model {} ...".format(idx+1, model_))
                    print ("Selected story prompt: {}".format(i+1))
                    start_time = time.time()
                    generated_text = text_generator(model_type=model_type, model_name_or_path=model_, prompt=prompt, padding_text=story[:50], xlm_lang="", length=length, temperature=temperature, top_k=top_k, top_p=0.9, no_cuda=no_cuda_bool, seed=42, stop_token=stop_token, verbose=False)
                    time_elapsed = time.time() - start_time
                    temp_pc = int(temperature*100)
                    filename_ = "gentext_"+model_+"_T"+str(temp_pc)+"_k"+str(top_k)+"_"+str(i)+".txt"
                    with open(os.path.join(save_dir, filename_),'w') as w_f:
                        w_f.write("Prompt: " + prompt + "\n")
                        w_f.write("Original: " + story + "\n")
                        w_f.write("Generated: " + generated_text + "\n")
                        w_f.write("Time elapsed: " + str(time_elapsed) + "\n")
    ##### get the directory of the samples by each model --done
    ##### read the files and get the dataframe from each model 
    if evaluate_bool:
        print ("Evaluation for {} model: ".format(eval_modelname))
        eval_modelObj = EvalDQ(eval_direcpath)
        print ("Reading the samples ...")    
        
        if eval_modelname == "fusion":
            df_modelObj = eval_modelObj.read_fusion_output()
        else:
            df_modelObj = eval_modelObj.read_data_strings()
            # print (df_modelObj["temp"].tolist())
            # exit()
        
        temp = set(df_modelObj["temp"].tolist())
        topK = set(df_modelObj["topK"].tolist())
        print ("The shape of the Dataframe object for model {} is {}:".format(eval_modelname, df_modelObj.shape))
        print ("The temperature and k values are: {} and {}:".format(temp, topK))
        
        if eval_RS:
            print ("Calculating the Readability scores ... ")
            print ("For the original stories ...")
            df_modelObj_RS_original = eval_modelObj.get_readability_scores(df_modelObj,"original")
            print ("The mean reading score values for the original files ...")
            print (df_modelObj_RS_original.mean(axis=0))
            print ("For the generated stories ...")
            df_modelObj_RS_generated = eval_modelObj.get_readability_scores(df_modelObj,"generated")
            print ("The mean reading score values for the generated files ...")
            print (df_modelObj_RS_generated.mean(axis=0))
        
        if eval_CW:
            print ("Calculating the percentage of content words VS stop words ...")
            print ("For the original stories ...")
            cw_ct_ori, sw_ct_ori = eval_modelObj.count_contentwords(df_modelObj, "original")
            mean_cw_ct_ori = statistics.mean(cw_ct_ori) #look at the normalized mean  
            mean_sw_ct_ori = statistics.mean(sw_ct_ori)
            print ("The normalized mean for content words is {} and for stop words is {}".format(mean_cw_ct_ori, mean_sw_ct_ori))
            print ("For the generated stories ...")
            cw_ct_gen, sw_ct_gen = eval_modelObj.count_contentwords(df_modelObj, "generated")
            mean_cw_ct_gen = statistics.mean(cw_ct_gen) #look at the normalized mean  
            mean_sw_ct_gen = statistics.mean(sw_ct_gen)
            print ("The normalized mean for content words is {} and for stop words is {}".format(mean_cw_ct_gen, mean_sw_ct_gen))

        if eval_NG:
            print ("Calculating the Story Prompt Relatedness scores ... ")
            print ("Calculating the average n-gram overlap with the prompt...")
            # avg_ngmoverlap_pc_gen = eval_modelObj.ngram_overlap(df_modelObj, ("generated", "prompt"), n=3)
            # print ("The average overlap percentage is {}".format(avg_ngmoverlap_pc_gen))
            print ("For the original stories ...")
            for i in [1,2,3]:
                print ("Getting the average for n={}".format(i))
                avg_ngmoverlap_pc_ori = eval_modelObj.ngram_overlap(df_modelObj, ("original", "prompt"), n=i, sw=sw, stem=stem)
                print ("The average overlap percentage is {}".format(avg_ngmoverlap_pc_ori))
            print ("For the generated stories ...")
            for i in [1,2,3]:
                print ("Getting the average for n={}".format(i))
                avg_ngmoverlap_pc_gen = eval_modelObj.ngram_overlap(df_modelObj, ("generated", "prompt"), n=i, sw=sw, stem=stem)
                print ("The average overlap percentage is {}".format(avg_ngmoverlap_pc_gen))

        if eval_PS:
            print ("Calculating the constituency parsing scores ...")
            print ("For the original stories ...")
            _, skew_scores_ori, kurt_scores_ori = eval_modelObj.parsing_score_calculation(df_modelObj, "original")
            mean_skew_scores_ori = statistics.mean(skew_scores_ori) #look at the normalized mean  
            mean_kurt_scores_ori = statistics.mean(kurt_scores_ori)
            print ("The mean skewness is {} and kurtosis is {}".format(mean_skew_scores_ori, mean_kurt_scores_ori))
            print ("For the generated stories ...")
            _, skew_scores_gen, kurt_scores_gen = eval_modelObj.parsing_score_calculation(df_modelObj, "generated")
            mean_skew_scores_gen = statistics.mean(skew_scores_gen) #look at the normalized mean  
            mean_kurt_scores_gen = statistics.mean(kurt_scores_gen)
            print ("The mean skewness is {} and kurtosis is {}".format(mean_skew_scores_gen, mean_kurt_scores_gen))
        
        if eval_SE:
            print ("Calculating the Story Prompt Relatedness scores ... ")
            print ("Calculating the sentence embedding similarity with the prompt...")
            print ("For the original stories ...")
            avg_sentemb_sim_ori = eval_modelObj.word2vec_sentsim(df_modelObj, ("original", "prompt"))
            print ("The average sentence embedding similarity is {}".format(avg_sentemb_sim_ori))
            print ("For the generated stories ...")
            avg_sentemb_sim_gen = eval_modelObj.word2vec_sentsim(df_modelObj, ("generated", "prompt"))
            print ("The average sentence embedding similarity is {}".format(avg_sentemb_sim_gen))

        if eval_SL:
            print ("Calculating the average sentence length ...")
            print ("For the orginal stories ...")
            sentlen_list_ori = eval_modelObj.average_sentence_length(df_modelObj, "original")
            mean_sentlen_ori = statistics.mean(sentlen_list_ori)
            print ("The average sentence length is {}".format(mean_sentlen_ori))
            print ("For the generated stories ...")
            sentlen_list_gen = eval_modelObj.average_sentence_length(df_modelObj, "generated")
            mean_sentlen_gen = statistics.mean(sentlen_list_gen)
            print ("The average sentence length is {}".format(mean_sentlen_gen))
        
        if eval_PF:
            print ("Calculating the POS tag frequency tag distribution ...")
            print ("For the original stories ...")
            df_modelObj_POS_ori = eval_modelObj.pos_tag_freqdist(df_modelObj, "original")
            print ("The mean POS tag percentages for the original files ...")
            POS_dict_ori = (df_modelObj_POS_ori.mean(axis=0)).to_dict()
            print ("NOUN: {} and VERB: {}".format(POS_dict_ori['NOUN']*100, POS_dict_ori['VERB']*100))
            print ("For the generated stories ...")
            df_modelObj_POS_gen = eval_modelObj.pos_tag_freqdist(df_modelObj, "generated")
            print ("The mean POS tag percentages for the generated files ...")
            POS_dict_gen = df_modelObj_POS_gen.mean(axis=0)
            print ("NOUN: {} and VERB: {}".format(POS_dict_gen['NOUN']*100, POS_dict_gen['VERB']*100))

        if eval_RW:
            print ("Calculating the rare word usage metrics ...")
            print ("For the generated stories ...")
            mean_ug_prblst_ori = eval_modelObj.get_rareword_usage(df_modelObj)
            mean_ug_ori = statistics.mean(mean_ug_prblst_ori)
            print ("The average unigram probability is {}".format(mean_ug_ori))
            

if __name__ == "__main__":
    main()