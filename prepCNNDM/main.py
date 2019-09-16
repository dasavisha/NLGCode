from evalData import EvalDQ
import os
import sys


def main():
    """
    give the path to the directory with the documents
    """
    filepath = sys.argv[1]    
    eDQ = EvalDQ(filepath)
    

if __name__ == "__main__":
    main()