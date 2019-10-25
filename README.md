# NLGCode
Repository to add all the Language Generation Models developed as part of thesis
This is the README File
---------------------------

1. Download the "writing prompts" dataset and untar the file
    --- mkdir stories
    --- cd stories
    --- wget https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar xvzf -

2. First round of preprocessing the data 
    --- Instead of using the full released dataset we limit to the first 1000 words
    --- An example code that trims the dataset to the first 1000 words of each story
    --- python main.py --filepath ../stories_data/writingPrompts/ --truncate 

3. Get the stats for the data
    --- Prints the number of articles and vocabulary count in each text
    --- python main.py --filepath ../stories_data/writingPrompts/ --stats
            {'train': 272600, 'test': 15138, 'valid': 15620} {'train': 272600, 'test': 15138, 'valid': 15620}
    --- python main.py --filepath ../stories_data/writingPrompts/ --count_vocab 
            The vocabulary for the stories in train is: 454219
            The vocabulary for the prompts in train is: 39892
            The vocabulary for the stories in test is: 83862
            The vocabulary for the prompts in test is: 11162
            The vocabulary for the stories in valid is: 85576
            The vocabulary for the prompts in valid is: 11129

4. Get the statistics as shown in Fan et. al. Table 1
    --- cd stories_data/writingPrompts
    --- wc -w [train/test/valid].wp_target 
            (160985243/8966916/9146281)
    --- wc -w [train/test/valid].wp_source 
            (7735772/425521/453716)

5. Running the pre-trained models
    --- python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples <N> --temperature <T> --top_k <k> --length <L>
        N = # of samples to generate using prompts from test set
        T = The softmax temperature ranges between 0 to 1 in float. Default value is 1.0.
        k = The integer value for top K sampling. Default is 0.
        L = The number of words to be generated in the sample. Default is 150. 
        
    --- OpenAI's GPT models and Google/CMU's Transformer-XL and XLNet are available at: https://huggingface.co/transformers/pretrained_models.html
        python run_generation.py --model_type=<> --length=150 --model_name_or_path=<> 
        
        --- GPT models
                a. openai-gpt, 110M parameters.
                    model_type = openai-gpt 
                    model_name_or_path = openai-gpt
                b. gpt2, 117M parameters.
                    model_type = gpt2 
                    model_name_or_path = gpt2
                c. gpt2-medium, 345M parameters.
                    model_type = gpt2 
                    model_name_or_path = gpt2-medium
                d. gpt2-large, 774M parameters.
                    model_type = gpt2 
                    model_name_or_path = gpt2-large
        --- Transformer models
                a. transfo-xl-wt103, 257M parameters.
                    model_type = transfo-xl 
                    model_name_or_path = transfo-xl-wt103 
                b  xlnet-base-cased, 110M parameters.
                    model_type = xlnet
                    model_name_or_path = xlnet-base-cased
                c. xlnet-large-cased, 340M parameters.
                    model_type = xlnet
                    model_name_or_path = xlnet-large-cased
    --- Conventional Seq2Seq models
        --- Fusion model
            --- Binarize the dataset
                <<add the command>>
            --- Run the generation
                fairseq-generate data-bin/writingPrompts --path /home/avisha/Desktop/Fall_2019/NLGCode/prepCNNDM/models/fusion_checkpoint.pt --batch-size 32 --beam 1 --sampling --sampling-topk 10 --temperature 0.8 --nbest 1 --model-overrides "{'pretrained_checkpoint':'/home/avisha/Desktop/Fall_2019/NLGCode/prepCNNDM/models/pretrained_checkpoint.pt'}"

6. Outputs saved in the 'save' directory
    --- Sample_R2: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 10 <<completed>>
    --- Sample_R3: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --save_dir "../save/Sample_R1" (Changes: the random sample numbers are not duplicates. The original story now has the entire content.) **server <<completed>> 
    --- Sample_R4: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --temperature 0.5 --save_dir "../save/Sample_R4" <<completed>>
    --- Sample_R5: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --temperature 0.75 --save_dir "../save/Sample_R5" <<completed>>
    --- Sample_R6: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --top_k 5 --save_dir "../save/Sample_R6" **server <<completed>>
    --- Sample_R7: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --top_k 10 --save_dir "../save/Sample_R7" **server <<unfinished>> -- 132 
    --- ##Sample_R8: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --top_k 50 --save_dir "../save/Sample_R8"
    --- Sample_R9: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --top_k 150 --save_dir "../save/Sample_R9" <<completed>>
    --- Sample_R10: python main.py --filepath ../stories_data/writingPrompts/ --generate --num_samples 200 --top_k 1000 --save_dir "../save/Sample_R10" <<completed>>


-------------------------------------------------------------------------------------------------------------------------------------------------------
7. Fine-tuning OpenAI's GPT models for the WritingPrompts dataset
    --- python run_lm_finetuning.py --output_dir=output_ft_gpt2 --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../stories_data/writingPrompts/train.wp_target --do_eval --eval_data_file=../stories_data/writingPrompts/test.wp_target --block_size=128 **server <<currently running>>
    --- python run_lm_finetuning.py --output_dir=output_ft_gpt2_med --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../stories_data/writingPrompts/train.wp_target --do_eval --eval_data_file=../stories_data/writingPrompts/test.wp_target **server <<aborted>>
    --- python run_lm_finetuning.py --output_dir=output_ft_gpt2 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../stories_data/writingPrompts/train.wp_target --do_eval --eval_data_file=../stories_data/writingPrompts/test.wp_target --block_size=128

    --- To speed up the process of fine tuning, we use 30% of the training dataset for the process.
        --- Extract X% of the samples in the training dataset
        --- Train the gpt2 and gpt2-medium on the extracted X% 
        --- 

    --- python run_lm_finetuning.py --output_dir=output_ft_gpt2 --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../stories_data/writingPrompts/valid.wp_target --do_eval --eval_data_file=../stories_data/writingPrompts/test.wp_target --block_size=128
        
        10/12/2019 16:09:42 - INFO - __main__ -   Saving features into cached file ../stories_data/writingPrompts/cached_lm_128_test.wp_target
        10/12/2019 16:09:43 - INFO - __main__ -   ***** Running evaluation  *****
        10/12/2019 16:09:43 - INFO - __main__ -     Num examples = 85176
        10/12/2019 16:09:43 - INFO - __main__ -     Batch size = 16
        Evaluating: 100%  5324/5324 [1:06:06<00:00,  1.34it/s]
        10/12/2019 17:15:49 - INFO - __main__ -   ***** Eval results  *****
        10/12/2019 17:15:49 - INFO - __main__ -     perplexity = tensor(21.2416)

    --- python run_lm_finetuning.py --output_dir=output_ft_gpt2_medium --model_type=gpt2 --model_name_or_path=gpt2-medium --do_train --train_data_file=../stories_data/writingPrompts/valid.wp_target --do_eval --eval_data_file=../stories_data/writingPrompts/test.wp_target --block_size=512

-------------------------------------------------------------------------------------------------------------------------------------------------------

8. Evaluating the output samples    
    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --reading_scores **
    
    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --content_words **

    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --ngram_overlap **

    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --sent_length **

    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --parse_scores 

    --- python main.py --filepath ../stories_data/writingPrompts/ --evaluate --eval_dir "../save/Sample_R1/Sample_R1/gpt2" --eval_model "gpt2" --pos_tag_fqd **
    



        
