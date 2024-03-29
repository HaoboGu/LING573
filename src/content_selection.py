import src.data_preprocessing as dp
import src.feature_calc as fc
import math
import numpy as np
import scipy as sp
import operator as op
import src.compression as comp

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
COMBINED = "Combined"

SENT_LEN_THRESHOLD = 10


class model:

    def __init__(self, name, idf, tagger, word_art=None, word_sum=None, art_mat=None, sum_mat=None):
        self.model_name = name
        self.idf = idf
        self.tagger = tagger
        self.klag = {}
        self.klga = {}
        if word_art is not None and word_sum is not None and art_mat is not None and sum_mat is not None:
            klag, klga = calculate_kl(word_art, word_sum, art_mat, sum_mat)
            self.klag = klag
            self.klga = klga


# This function gives the content selection result of a given docset.
# list[data_preprocessing.sentence] cs(list[data_preprocessing.docSet, float)
def cs(docset, compression_rate, model_type):
    allsentencelist = generate_sentencelist(docset, model_type.tagger)
    idf = model_type.idf
    docset_tokenDict = docset.tokenDict()
    model_name = model_type.model_name
    if op.eq(model_name, LEXRANK):
        sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict, idf)
        sentence_similarity = calculate_sentence_similarity(sentence_matrix)
        important_score_vector = LexRank(sentence_similarity)

    elif op.eq(model_name, KL_DIVERGENCE):
        new_sentence_list = replace_sentence_token_dict(allsentencelist, model_type)
        # sentence_matrix = sentence2matrix(new_sentence_list, docset_tokenDict)
        # Q, R, P = sp.linalg.qr(sentence_matrix)
        # important_score_vector = kl_score(sentence_matrix)
        important_score_vector = get_sentence_score(new_sentence_list)

    elif op.eq(model_name, COMBINED):
        sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict, idf)
        # check_metrix(sentence_matrix)
        sentence_similarity = calculate_sentence_similarity(sentence_matrix)
        important_score_vector1 = LexRank(sentence_similarity)
        new_sentence_list = replace_sentence_token_dict(allsentencelist, model_type)
        # sentence_matrix = sentence2matrix(new_sentence_list, docset_tokenDict)
        # Q, R, P = sp.linalg.qr(sentence_matrix)
        # important_score_vector = kl_score(sentence_matrix)
        important_score_vector2 = get_sentence_score(new_sentence_list)
        # important_score_vector2 = smooth(important_score_vector2)
        important_score_vector = combine_two_method(important_score_vector1, important_score_vector2)

    output_sen_num = int(len(allsentencelist) * compression_rate)
    important_sentences = generate_most_important_sentences(important_score_vector, allsentencelist, output_sen_num)
    return important_sentences

'''
def test_ave_score(docset, model_type):
    allsentencelist = generate_sentencelist(docset)
    docset_tokenDict = docset.tokenDict()
    sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict)
    sentence_similarity = calculate_sentence_similarity(sentence_matrix)
    important_score_vector1 = LexRank(sentence_similarity)
    new_sentence_list = replace_sentence_token_dict(allsentencelist, model_type)
    important_score_vector2 = get_sentence_score(new_sentence_list)
    sum1 = sum(important_score_vector1)
    sum2 = sum(important_score_vector2)
    count = len(important_score_vector1)
    return sum1, sum2, count
'''


def combine_two_method(score_list1, score_list2):
    score = []
    for i in range(len(score_list1)):
        score.append(score_list1[i] * 0.25 + score_list2[i] * 0.75)
    return score


# This function generates a sentence list of all sentences in a given docset.
# list[data_preprocessing.sentence] generate_sentencelist(data_preprocesing.docSet)
def generate_sentencelist(docset, tagger):
    docCluster = docset.documentCluster()
    sentencelist = []
    for doc in docCluster:
        sentences = doc.sentences()
        for sent in sentences:
            sent = comp.compress_sent(sent, tagger)
            if sent.length() > 0:
                sentencelist.append(sent)
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
def sentence2matrix(sentences, tokenDict, idf_dict):
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
            tf = float(value) / sent.length()
            idf = idf_dict[key]
            sentence_matrix[i][ind] = tf * idf
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

    return normalize_score(to_vector)


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
            if math.isnan(matrix[i][j]) or math.isnan(sim_sum) or sim_sum == 0:
                print()
            tranmatrix[i][j] = matrix[i][j] / sim_sum
    return tranmatrix


def converging(vector1, vector2, difference=converge_standard):
    length = len(vector1)
    for i in range(length):
        if math.fabs(vector1[i] - vector2[i]) > difference:
            return False
    return True


def generate_most_important_sentences(score_list, sentence_list, num):
    apply_threshold(score_list, sentence_list)
    maxn = get_max_n(score_list, num)
    result = []
    for item in maxn:
        score = item[1]
        sent = sentence_list[item[0]]
        sent._score = score
        result.append(sent)
    return result


def apply_threshold(score_list, sentence_list):
    for i in range(len(sentence_list)):
        sentence = sentence_list[i]
        if sentence.length() < SENT_LEN_THRESHOLD:
            score_list[i] = -2


def normalize_score(score_list):
    biggest_score = get_max_score(score_list)[1]
    smallest_score = get_min_score(score_list)[1]
    for i in range(len(score_list)):
        if score_list[i] > 0:
            score_list[i] = score_list[i] / biggest_score
        else:
            score_list[i] = - score_list[i] / smallest_score
    return score_list


def smooth(score_list):
    for i in range(len(score_list)):
        if score_list[i] > 0:
            score_list[i] = score_list[i] * (2 - 1 * score_list[i])
        else:
            score_list[i] = score_list[i] * (2 + 1 * score_list[i])
    return score_list


def get_max_n(score_list, n):
    topn = n
    if n > len(score_list)-1:
        topn = len(score_list)-1
    result = []

    for i in range(topn):
        result.append([i, score_list[i]])
    smallest = get_min(result)
    for i in range(topn, len(score_list)-1):
        if score_list[i] > smallest[1]:
            result[smallest[0]] = [i, score_list[i]]
            smallest = get_min(result)
    return result


def get_max_score(score_list):
    biggest = score_list[0]
    index = 0
    for i in range(1, len(score_list)):
        if score_list[i] > biggest:
            biggest = score_list[i]
            index = i
    return [index, biggest]


def get_min_score(score_list):
    smallest = score_list[0]
    index = 0
    for i in range(1, len(score_list)):
        if score_list[i] < smallest:
            smallest = score_list[i]
            index = i
    return [index, smallest]


def get_min(score_list):
    smallest = score_list[0][1]
    index = 0
    for i in range(1, len(score_list)):
        if score_list[i][1] < smallest:
            smallest = score_list[i][1]
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
        amount = 0
        for token in token_dict.keys():
            ag = 0
            ga = 0
            inag = token in kl_ag
            inga = token in kl_ga
            if inag:
                ag = kl_ag[token]
            if inga:
                ga = kl_ga[token]
            score = (ga - ag) / 2.0
            new_token_dict[token] = score
            if inag and inga:
                amount += 1
        '''
        if amount == 0:
            score = 0
        else:
            score = sum(new_token_dict.values()) / amount
        '''
        score = sum(new_token_dict.values()) / len(new_token_dict)
        # score = sum(new_token_dict.values())
        new_sent = dp.sentence(sentence.idCode(), sentence.content(), sentence.index(), score,
                               sentence.length(), new_token_dict, sentence.doctime())
        new_sent_list.append(new_sent)
    return new_sent_list


def get_sentence_score(sentence_list):
    score_list = []
    for sentence in sentence_list:
        score_list.append(sentence.score())
    return normalize_score(score_list)


def get_idf(corp):
    df = {}
    corpus_token_dict = corp.tokenDict().keys()
    doc_number = 0
    for token in corpus_token_dict:
        df[token] = 0
    for docset in corp.docsetList():
        for doc in docset.documentCluster():
            doc_tokens = doc.tokenDict()
            for tokenid in df.keys():
                if tokenid in doc_tokens:
                    df[tokenid] += 1
                    continue
            doc_number += 1
    for tokenid in df.keys():
        df[tokenid] = math.log(float(doc_number) / df[tokenid])
    return df


def train_model(corpus, model_type, tagger):
    docsetlist = corpus.docsetList()
    idf = get_idf(corpus)
    if op.eq(model_type, LEXRANK):
        return model(model_type, idf, tagger)
    elif op.eq(model_type, KL_DIVERGENCE):
        word_list_art, word_list_sum, art_matrix, sum_matrix = fc.feature_weight_calc(docsetlist)
        return model(model_type, idf, tagger, word_list_art, word_list_sum, art_matrix, sum_matrix)
    elif op.eq(model_type, COMBINED):
        word_list_art, word_list_sum, art_matrix, sum_matrix = fc.feature_weight_calc(docsetlist)
        return model(model_type, idf, tagger, word_list_art, word_list_sum, art_matrix, sum_matrix)


def check_metrix(m):
    for i in range(len(m)):
        if sum(m[i]) == 0:
            print()
        for j in range(len(m[i])):
            if math.isnan(m[i][j]):
                print()


if __name__ == "__main__":
    type1 = "LexRank"
    type2 = "KL_Divergence"
    type3 = "Combined"
    compress_corpus = data_home + "/other_resources/compression_corpus"
    tagger = comp.create_a_tagger(compress_corpus)
    training_corpus = dp.generate_corpus(demo_training_corpus_file, aqua, aqua2, human_judge)
    docsetlist = training_corpus.docsetList()
    cs_model = train_model(training_corpus, type3, tagger)

    for docset in docsetlist:
        important_sentences = cs(docset, comprate, cs_model)
        for sentence in important_sentences:
            print(sentence.content())
        print("\n")
    '''
    s1 = 0
    s2 = 0
    s3 = 0
    for docset in docsetlist:
        sum1, sum2, count = test_ave_score(docset, cs_model)
        s1 += sum1
        s2 += sum2
        s3 += count
    print(str(s1/s3))
    print(str(s2/s3))
    '''
