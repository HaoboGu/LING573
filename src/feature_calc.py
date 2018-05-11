import src.data_preprocessing as dp
import numpy as np
import scipy as sp
import nltk
import math

training_corpus_file = "Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
demo_training_corpus_file = "test.xml"
aqua = "LDC08T31"
aqua2 = "LDC08T25/data"
human_judge = "Data/models/training/2009"

def process_stopwords_punctuation(tokens):
    stopWords = set(nltk.corpus.stopwords.words('english'))
    new_tokens = list()
    for tok in tokens:
        if tok in stopWords:
            continue
        elif tok.isalpha() == False:
            continue
        else:
            new_tokens.append(tok.lower())
    return new_tokens

def add_word_to_collection(new_tokens, word_set, word_dict):
    for word in new_tokens:
        word_set.add(word)
        if hash(word) in word_dict:
            word_dict[hash(word)] += 1
        else:
            word_dict[hash(word)] = 1

def feature_weight_calc(docsetlist):
    word_set_art = set()
    word_set_sum = set()
    word_dic_article = dict()
    word_dic_summary = dict()
    word_count_article = 0
    word_count_summary = 0
    for docset in docsetlist:
        for summary in docset._humanSummary:
            summary_tokens = nltk.tokenize.word_tokenize(summary)
            new_summary_tokens = process_stopwords_punctuation(summary_tokens)
            word_count_summary += len(new_summary_tokens)
            add_word_to_collection(new_summary_tokens, word_set_sum, word_dic_summary)
        for doc in docset._documentCluster:
            for sent in doc._sentences:
                tokens = nltk.tokenize.word_tokenize(sent._content)
                new_tokens = process_stopwords_punctuation(tokens)
                word_count_article += len(new_tokens)
                add_word_to_collection(new_tokens, word_set_art, word_dic_article)
    
    word_list_art = list(word_set_art)
    word_list_sum = list(word_set_sum)

    art_len = len(word_list_art)
    sum_len = len(word_list_sum)
                
    art_matrix = np.zeros(art_len)
    sum_matrix = np.zeros(sum_len)

    for i in range(0, art_len):
        art_matrix[i] = math.log(word_dic_article[hash(word_list_art[i])]) - math.log(word_count_article)
    for j in range(0, sum_len):
        sum_matrix[j] = math.log(word_dic_summary[hash(word_list_sum[j])]) - math.log(word_count_summary)
    
    return word_list_art, word_list_sum, art_matrix, sum_matrix


if __name__ == "__main__":
    training_corpus = dp.generate_corpus(training_corpus_file, aqua, aqua2, human_judge)
    docsetlist = training_corpus.docsetList()
    word_list_art, word_list_sum, art_matrix, sum_matrix = feature_weight_calc(docsetlist)


        
