#!/usr/bin/env python3

### imports
## modules
import src.data_preprocessing as dp
import src.content_selection as cs

## other packages
import operator
import nltk
import numpy as np
from scipy.spatial import distance
import os
import sys
import re



def sort_sentence_list(important_sentences):
    """
    ========================
    DESCRIPTION: this function takes a list of sentence objects as input
    and sort them in terms of publication date in recency and within-article indices.
    ========================
    INPUT: important_sentences: list[sentence]
    ========================
    OUTPUT: sent_list: list[sentence]
    ========================
    """
    id_dic = {} # initialize a dictionary with idCode as keys and doctime as val
    id_dic2 = {} # initialize another dictionary with idCode as keys
    for s in important_sentences: # traverse through all sents
        if s.idCode() not in id_dic: # fill in dicts
            id_dic[s.idCode()] = s.doctime()
            id_dic2[s.idCode()] = {}
        id_dic2[s.idCode()][s.index()] = s # group sents w.r.t. idCode
      
    sorted_id_dic = sorted(id_dic.items(), key=operator.itemgetter(1)) # sort id_dic
    
    #word_count = 0 # initialize count for words of summary, capped at 100
    sent_list = [] # initialize a list of sents for output
    for i in sorted_id_dic: 
        sent_id = i[0]
        for j in id_dic2:
            if j == sent_id: # first sort sents w.r.t. publication date
                ind_list = list(id_dic2[j].keys())
                ind_list_sorted = sorted(ind_list)
                for m in ind_list_sorted: # second sort sents w.r.t. index
                    sent = id_dic2[j][m]
                    sent_list.append(sent)
    return(sent_list)


def write_to_file(docset,sent_list,output):
    """
    ========================
    DESCRIPTION: this function takes a sorted list of sentences and write it to output files
    ========================
    INPUT: sent_list: list[sentence]
    ========================
    OUTPUT: None
    ========================
    """
    with open(str(docset.idCode())+'.'+output,'w') as f1:
        for s in sent_list:
            f1.write(re.sub(r'\n',' ',s.content())+os.linesep)

            

def io_wrapper(training_corpus_file,aqua,aqua2,human_judge,output):
    """
    ========================
    DESCRIPTION: this function is a wrapper which loads and process the training documents, extract important sentences,
    and sort the sentences in order.
    ========================
    """
    # read in and preprocess documents
    sys.stderr.write("Start preprocessing documents ...\n")
    training_corpus = dp.generate_corpus(training_corpus_file,aqua,aqua2,human_judge)
    sys.stderr.write("--- Finished preprocessing documents ......\n")
    
    # extract sentences
    docsetlist = training_corpus.docsetList()
    docset_dic = {}
    for docset in docsetlist: # traverse through all document sets
        if docset.idCode() not in docset_dic:
            docset_dic[docset.idCode()] = []
        sys.stderr.write("Start extracting important sentences for docset: "+str(docset.idCode())+"\n")
        important_sentences = cs.cs(docset, compression_rate=cs.comprate) # content selection
        sent_list = sort_sentence_list(important_sentences) # sort important sentences
        docset_dic[docset.idCode()] = sent_list
        sys.stderr.write("--- Finished extracting important sentences ......\n")
        
        # write to output files
        sys.stderr.write("Start writing to file ...\n")
        write_to_file(docset,sent_list,output)
        sys.stderr.write("--- Finished writing to file ......\n")
    return(docset_dic)
    
    
    
            
            
if __name__ == "__main__":
    sys.stderr.write("------ Start ------\n")
    
    # test variable(s) for output
    output = sys.argv[1]
    
    # read in and preprocess documents
    """
    FIXME: Please modify the data directory (in content selection script) before running the following line of code.
    FIXME: This script assumes the same directory as other modules
    """
    docset_dic = io_wrapper(cs.demo_training_corpus_file,cs.aqua,cs.aqua2,cs.human_judge,output)
    
    sys.stderr.write("------ Finished ------\n")
            
            
        




