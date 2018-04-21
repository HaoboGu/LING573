import src.data_preprocessing as dp
import math
import numpy as np
import scipy as sp

data_home = "/Users/user/Documents/GitHub/573data/573"
training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
demo_training_corpus_file = data_home + "/Data/Documents/training/UpdateSumm09_demo_test_topics.xml"
aqua = data_home + "/AQUAINT"
aqua2 = data_home + "/AQUAINT-2/data"
human_judge = data_home + "/Data/models/training/2009"
comprate = 0.1  # The number of sentence selected
converge_standard = 0.001  # Used to judge if the score is converging


# This function gives the content selection result of a given docset.
# list[data_preprocessing.sentence] cs(list[data_preprocessing.docSet, float)
def cs(docset, compression_rate):
    allsentencelist = generate_sentencelist(docset)
    docset_tokenDict = docset.tokenDict()
    sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict)
    sentence_similarity = calculate_sentence_similarity(sentence_matrix)
    importance_score_vector = LexRank(sentence_similarity)
    output_sen_num = int(len(allsentencelist) * compression_rate)
    importance_sentences = generate_most_important_sentences(importance_score_vector, allsentencelist, output_sen_num)
    return importance_sentences


# This function generates a sentence list of all sentences in a given docset.
# list[data_preprocessing.sentence] generate_sentencelist(data_preprocesing.docSet)
def generate_sentencelist(docset):
    docCluster = docset.documentCluster()
    sentencelist = []
    for doc in docCluster:
        sentences = doc.sentences()
        for sentence in sentences:
            if len(sentence._tokenDict) > 0:
                sentencelist.append(sentence)
    return sentencelist


# This function calculate similarity of every two sentences, given the matrix of these sentences
# The distance function is cosine distance, the bigger, the more similar
# matrix alculate_sentence_similarity(matrix)
def calculate_sentence_similarity(sent_matrix, threshold=0.0):
    sent_len = len(sent_matrix)
    sim_matrix = np.zeros((sent_len, sent_len))
    for i in range(sent_len):
        for j in range(sent_len):
            sent_i = sent_matrix[i]
            sent_j = sent_matrix[j]
            sent_sim = 1 - sp.spatial.distance.cosine(sent_i, sent_j)
            if sent_sim > threshold:
                sim_matrix[i][j] = sent_sim
    return sim_matrix


# This function transfers all sentences in a docset into a sentence matrix
def sentence2matrix(sentences, tokenDict):
    hash2index = {}
    index2hash = list(tokenDict.keys())
    tokenNum = len(index2hash)
    sentenceNum = len(sentences)
    sentence_matrix = np.zeros((sentenceNum, tokenNum))
    for i in range(tokenNum):
        hash2index[index2hash[i]] = i
    for i in range(sentenceNum):
        sent = sentences[i]
        sen_tokenDict = sent.tokenDict()
        for key, value in sen_tokenDict.items():
            if key not in hash2index:
                continue
            ind = hash2index[key]
            sentence_matrix[i][ind] = value
    return sentence_matrix


def LexRank(simi_matrix):
    length = len(simi_matrix)
    from_vector = np.ones(length)
    tranprob_matrix = matrix2tranmatrix(simi_matrix)
    T_tranprob_matrix = tranprob_matrix.transpose()
    to_vector = np.matmul(T_tranprob_matrix, from_vector)
    while not converging(from_vector, to_vector):
        from_vector = to_vector
        to_vector = np.matmul(T_tranprob_matrix, from_vector)

    return to_vector


# This function transfer a matrix into a transition probability matrix.
# matrix matrix2tranmatrix(matrix)
def matrix2tranmatrix(matrix):
    size = len(matrix)
    tranmatrix = np.zeros((size, size))
    for i in range(size):
        sim_sum = matrix[i].sum()
        for j in range(size):
            tranmatrix[i][j] = matrix[i][j] / sim_sum
    return tranmatrix


def converging(vector1, vector2, difference=converge_standard):
    length = len(vector1)
    for i in range(length):
        if math.fabs(vector1[i] - vector2[i]) > difference:
            return False
    return True


def generate_most_important_sentences(score_list, sentence_list, num):
    maxn = get_max_n(score_list, num)
    result = []
    for item in maxn:
        score = item[1]
        sent = sentence_list[item[0]]
        sent._score = score
        result.append(sent)
    return result


def get_max_n(list, n):
    topn = n
    if n > len(list)-1:
        topn = len(list)-1
    result = []

    for i in range(topn):
        result.append([i, list[i]])
    smallest = get_min(result)
    for i in range(topn, len(list)-1):
        if list[i] > smallest[1]:
            result[smallest[0]] = [i, list[i]]
            smallest = get_min(result)
    return result


def get_min(list):
    smallest = list[0][1]
    index = 0
    for i in range(1, len(list)):
        if list[i][1] < smallest:
            smallest = list[i][1]
            index = i
    return [index, smallest]


if __name__ == "__main__":

    training_corpus = dp.generate_corpus(demo_training_corpus_file, aqua, aqua2, human_judge)
    docsetlist = training_corpus.docsetList()
    for docset in docsetlist:
        important_sentences = cs(docset, compression_rate=comprate)
        for sentence in important_sentences:
            print(sentence.content())
        print("\n")
