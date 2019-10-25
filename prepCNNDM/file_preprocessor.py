import os
import sys
import collections
import spacy

nlp = spacy.load("en_core_web_lg")


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
VOCAB_SIZE = 200000
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

class FilePreprocessor(object):

    def __init__(self, directory):
        self.stories_dir = directory
        self.data = ["train", "test", "valid"]

    def truncate_stories(self, num_words=1000):
        """trims the dataset to the threshold of number of given words
        """
        # ctr = 0
        for name in self.data:
            with open(os.path.join(self.stories_dir, name+".wp_target")) as f: #read from the file
                stories = f.readlines()
            stories = [" ".join(i.split()[:num_words]) for i in stories]
            print ("Done reading: {} ...".format(name))
            with open(os.path.join(self.stories_dir, name+".wp_target"), "w") as o: #write to the same file
                for line in stories:
                    o.write(line.strip() + "\n")
            print ("Done writing: {} ...".format(name))

    def check_num_stories(self):
        """**cross check the number of files in the given directory
        """
        num_stories = {}
        num_prompts = {}
        for name in self.data:
            with open(os.path.join(self.stories_dir, name+".wp_target")) as f_t: #read from the file
                stories = f_t.readlines()
                num_stories[name] = len(stories)
        
        for name in self.data:
            with open(os.path.join(self.stories_dir, name+".wp_source")) as f_s: #read from the file
                prompts = f_s.readlines()
                num_prompts[name] = len(prompts)
        return num_stories, num_prompts

    def read_text_file(self, text_file):
        """Read the given text file
        returns the lines as a list
        """
        lines = list()
        with open(text_file) as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def get_art_prp_file(self):
        """
        returns a dict of filename with a tuple 
        first element is the article and second element is the summary
        """
        story_files_dict = {}
        for name in self.data:
            story_file_path = os.path.join(self.stories_dir, name+".wp_target")
            prompt_file_path = os.path.join(self.stories_dir, name+".wp_source")
            story_lines = self.read_text_file(story_file_path)
            story_lines = [line.lower() for line in story_lines]
            prompt_lines = self.read_text_file(prompt_file_path)
            prompt_lines = [line.lower() for line in prompt_lines]
            story_files_dict[name] = list()
            for art, prp in zip(story_lines, prompt_lines):
                tup_ = (prp, art)
                story_files_dict[name].append(tup_)
        
        return story_files_dict
    
    def make_vocabulary(self):
        """
        Make the vocabulary file with the data
        """
        vocab_counter_story = {}
        vocab_counter_prompt = {}
        story_files_dict = self.get_art_prp_file() 
        for name in story_files_dict.keys():
            vocab_counter_story[name] = collections.Counter()
            vocab_counter_prompt[name] = collections.Counter()
            for idx, item in enumerate(story_files_dict[name]):
                if idx % 5000 == 0: print ("Processed files: {}".format(idx))
                ##vocab counter
                story_tokens = list()
                prompt_tokens = list()
                story = nlp(item[1])
                prompt = nlp(item[0])
                for token in story:
                    story_tokens.append(token.text)
                for token in prompt:
                    prompt_tokens.append(token.text)
                vocab_counter_story[name].update(story_tokens)
                vocab_counter_prompt[name].update(prompt_tokens)
            print ("The vocabulary for the stories in {} is: {}".format(name, len(vocab_counter_story[name].keys())))
            print ("The vocabulary for the prompts in {} is: {}".format(name, len(vocab_counter_prompt[name].keys())))
        
        # print ("Writing vocab file...")
        # with open(os.path.join(self.stories_dir, "vocab"), 'w') as writer:
        #     for word, count in vocab_counter.most_common(VOCAB_SIZE):
        #         writer.write(word + ' ' + str(count) + '\n')
        # print ("Finished writing vocab file")
        # return vocab_counter
        return vocab_counter_prompt, vocab_counter_story

