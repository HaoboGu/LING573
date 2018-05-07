import data_preprocessing as dp
import numpy as np
import scipy as sp
import nltk

def get_log_sum_fact(n, r):
    return_sum = 0
    for i in range(n-r+1,n+1):
        return_sum += np.log(i)
    return return_sum

def calculate_log_comb(n, r):
    upper = get_log_sum_fact(n, r)
    lower = get_log_sum_fact(r, r)
    return upper-lower

def num_log(n):
    if n == 0:
        return 0
    else:
        return np.log(n)

def calculate_llr(docsetlist):
    num_topics = len(docsetlist)
    word_list = list()
    topic_dict = dict()
    topic_list = list()
    topic_length_dict = dict()
    word_whole_dict = dict()
    for doceset in docsetlist:
        if doceset._topicID not in topic_list:
            topic_list.append(doceset._topicID)
        if doceset._topicID not in topic_dict:
            topic_dict[hash(doceset._topicID)] = dict()
        if doceset._topicID not in topic_length_dict:
            topic_length_dict[hash(doceset._topicID)] = 0
        for doc in doceset._documentCluster:
            for sent in doc._sentences:
                tokens = nltk.tokenize.word_tokenize(sent._content)
                topic_length_dict[hash(doceset._topicID)] += len(tokens)
                for tok in tokens:
                    if tok not in word_list:
                        word_list.append(tok)
                    if hash(tok) in topic_dict[hash(doceset._topicID)]:
                        topic_dict[hash(doceset._topicID)][hash(tok)] += 1
                    else:
                        topic_dict[hash(doceset._topicID)][hash(tok)] = 1
                    if hash(tok) in word_whole_dict:
                        word_whole_dict[hash(tok)] += 1
                    else:
                        word_whole_dict[hash(tok)] = 1
                                   
    num_words = len(word_list)
    llr_matrix = np.zeros((num_topics, num_words))

    n_0 = 0
    for topic_id in topic_list:
        n_0 +=  topic_length_dict[hash(topic_id)]

    for i, topic_id in enumerate(topic_list):
        n_t = topic_length_dict[hash(topic_id)]
        n_b = n_0 - n_t
        for j, word in enumerate(word_list):
            k_0 = word_whole_dict[hash(word)]
            if hash(word) in topic_dict[hash(topic_id)]:
                k_t = topic_dict[hash(topic_id)][hash(word)]
            else:
                k_t = 0
            k_b = k_0 - k_t
            p_b = k_b/n_b
            p_t = k_t/n_t
            p_0 = k_0/n_0
            alternative_in = get_log_sum_fact(n_t, k_t) + num_log(np.power(p_t, k_t)*np.power(1-p_t,n_t-k_t))
            alternative_out = get_log_sum_fact(n_b, k_b) + num_log(np.power(p_b, k_b)*np.power(1-p_b,n_b-k_b))
            null_in = get_log_sum_fact(n_t, k_t) + num_log(np.power(p_0, k_t)*np.power(1-p_0,n_t-k_t))
            null_out = get_log_sum_fact(n_b, k_b) + num_log(np.power(p_0, k_b)*np.power(1-p_0,n_b-k_b))
            final_llr = (null_in+null_out)-(alternative_in+alternative_out)
            llr_matrix[i][j] = final_llr
    return word_list, topic_list, llr_matrix
