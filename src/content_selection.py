import src.data_preprocessing as dp
import src.feature_calc as fc
import math
import numpy as np
import scipy as sp
import operator as op

data_home = "/Users/user/Documents/GitHub/573data/573"
training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
demo_training_corpus_file = data_home + "/Data/Documents/training/UpdateSumm09_demo_test_topics.xml"
aqua = data_home + "/AQUAINT"
aqua2 = data_home + "/AQUAINT-2/data"
human_judge = data_home + "/Data/models/training/2009"
comprate = 0.1  # The number of sentence selected
converge_standard = 0.001  # Used to judge if the score is converging

LEXRANK = "LexRank"
KL_DIVERGENCE = "KL_Divergence"


class model:

    def __init__(self, name, word_art=None, word_sum=None, art_mat=None, sum_mat=None):
        self.model_name = name
        self.klag = {}
        self.klga = {}
        if word_art is not None and word_sum is not None and art_mat is not None and sum_mat is not None:
            klag, klga = calculate_kl(word_art, word_sum, art_mat, sum_mat)
            self.klag = klag
            self.klga = klga


# This function gives the content selection result of a given docset.
# list[data_preprocessing.sentence] cs(list[data_preprocessing.docSet, float)
def cs(docset, compression_rate, model_type):
    allsentencelist = generate_sentencelist(docset)
    docset_tokenDict = docset.tokenDict()
    model_name = model_type.model_name
    if op.eq(model_name, LEXRANK):
        sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict)
        sentence_similarity = calculate_sentence_similarity(sentence_matrix)
        important_score_vector = LexRank(sentence_similarity)

    elif op.eq(model_name, KL_DIVERGENCE):
        new_sentence_list = replace_sentence_token_dict(allsentencelist, model_type)
        sentence_matrix = sentence2matrix(new_sentence_list, docset_tokenDict)
        # Q, R, P = sp.linalg.qr(sentence_matrix)
        important_score_vector = kl_score(sentence_matrix)

    output_sen_num = int(len(allsentencelist) * compression_rate)
    important_sentences = generate_most_important_sentences(important_score_vector, allsentencelist, output_sen_num)
    return important_sentences


# This function generates a sentence list of all sentences in a given docset.
# list[data_preprocessing.sentence] generate_sentencelist(data_preprocesing.docSet)
def generate_sentencelist(docset):
    docCluster = docset.documentCluster()
    sentencelist = []
    for doc in docCluster:
        sentences = doc.sentences()
        for sentence in sentences:
            if len(sentence.tokenDict()) > 0:
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


def kl_score(sentence_matrix):
    score_list = []
    for i in range(len(sentence_matrix)):
        score_list.append(sum(sentence_matrix[i]))
    return score_list


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


def calculate_kl(art_list, sum_list, art_prob, sum_prob):
    art_prob_dic = {}
    sum_prob_dic = {}
    for i in range(len(art_list)):
        hash_art = hash(art_list[i])
        if hash_art not in art_prob_dic:
            art_prob_dic[hash_art] = art_prob[i]

    for i in range(len(sum_list)):
        hash_sum = hash(sum_list[i])
        if hash_sum not in sum_prob_dic:
            sum_prob_dic[hash_sum] = sum_prob[i]

    kl_ag = {}
    kl_ga = {}

    for key, value in art_prob_dic.items():
        lg_pra = value
        if key not in sum_prob_dic:
            kl_ga[key] = 0.0
        else:
            lg_prg = sum_prob_dic[key]
            prg = math.exp(lg_prg)
            kl_ga[key] = prg * (lg_prg - lg_pra)

    for key, value in sum_prob_dic.items():
        lg_prg = value
        if key not in art_prob_dic:
            kl_ag[key] = 0.0
        else:
            lg_pra = art_prob_dic[key]
            pra = math.exp(lg_pra)
            kl_ag[key] = pra * (lg_pra - lg_prg)

    return kl_ag, kl_ga


def replace_sentence_token_dict(sentence_list, model_type):
    new_sent_list = []
    kl_ag = model_type.klag
    kl_ga = model_type.klga
    for i in range(len(sentence_list)):
        sentence = sentence_list[i]
        token_dict = sentence.tokenDict()
        new_token_dict = {}
        for token in token_dict.keys():
            ag = 0.0
            ga = 0.0
            if token in kl_ag:
                ag = kl_ag[token]
            if token in kl_ga:
                ga = kl_ga[token]
            new_token_dict[token] = (ga - ag) / 2.0
        new_sent = dp.sentence(sentence.idCode(), sentence.content(), sentence.index(), sentence.score(),
                               sentence.length(), new_token_dict, sentence.doctime())
        new_sent_list.append(new_sent)
    return new_sent_list


def train_model(corpus, model_type):
    docsetlist = corpus.docsetList()
    if op.eq(model_type, LEXRANK):
        return model(model_type)
    elif op.eq(model_type, KL_DIVERGENCE):
        word_list_art, word_list_sum, art_matrix, sum_matrix = fc.feature_weight_calc(docsetlist)
        return model(model_type, word_list_art, word_list_sum, art_matrix, sum_matrix)


if __name__ == "__main__":
    type1 = "LexRank"
    type2 = "KL_Divergence"
    training_corpus = dp.generate_corpus(demo_training_corpus_file, aqua, aqua2, human_judge)
    docsetlist = training_corpus.docsetList()
    cs_model = train_model(training_corpus, type1)
    for docset in docsetlist:
        important_sentences = cs(docset, comprate, cs_model)
        for sentence in important_sentences:
            print(sentence.content())
        print("\n")
