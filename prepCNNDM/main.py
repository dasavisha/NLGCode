from evalData import EvalDQ
from file_preprocessor import FilePreprocessor
import os
import sys


def main():
    """
    give the path to the directory with the documents
    """
    filepath = sys.argv[1]
    f_prep = FilePreprocessor(filepath) 
    article_abstract_dict = f_prep.write_art_abs_file()   
    eDQ = EvalDQ(filepath)



if __name__ == "__main__":
    main()