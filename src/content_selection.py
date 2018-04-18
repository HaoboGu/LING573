import src.data_preprocessing as dp
import nltk

data_home = "/Users/user/Documents/GitHub/573data/573"
training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
aqua = data_home + "/AQUAINT"
aqua2 = data_home + "/AQUAINT-2/data"
human_judge = data_home + "/Data/models/training/2009"
comprate = 0.1


# This function gives the content selection result of a given docset.
# list[data_preprocessing.sentence] cs(list[data_preprocessing.docSet, float)
def cs(docset, compression_rate=comprate):
    allsentencelist = generate_sentencelist(docset)
    docset_tokenDict = docset.tokenDict()
    sentence_matrix = sentence2matrix(allsentencelist, docset_tokenDict)
    sentence_similarity = calculate_sentence_similarity(sentence_matrix)
    importance_score_vector = LexRank(sentence_similarity)
    importance_sentences = generate_most_important_sentences(importance_score_vector, allsentencelist)
    return importance_sentences


# This function generates a sentence list of all sentences in a given docset.
# list[data_preprocessing.sentence] generate_sentencelist(data_preprocesing.docSet)
def generate_sentencelist(docset):
    return []


# This function calculate similarity of every two sentences, given the matrix of these sentences
# matrix alculate_sentence_similarity(matrix)
def calculate_sentence_similarity(sent_matrix):

    return 0.0


# This function transfers all sentences in a docset into a sentence matrix
def sentence2matrix(sentences, tokenDict):

    return 0


def LexRank(simi_matrix):
    from_vector = generate_random_vector(length=1)
    tranprob_matrix = matrix2tranmatrix(simi_matrix)
    to_vector = tranprob_matrix * from_vector
    while not converging(from_vector, to_vector):
        from_vector = to_vector
        to_vector = tranprob_matrix * from_vector
    return to_vector

# This function transfer a matrix into a transition probability matrix.
# matrix matrix2tranmatrix(matrix)
def matrix2tranmatrix(matrix):
    return 0.0


# This function generate a vector of given length. The value of each dimension is between 0 and 1
def generate_random_vector(length):
    return 0.0


def converging(vector1, vector2, difference=0.001):
    return True


def generate_most_important_sentences(score_list, sentence_list):
    return []


if __name__ == "__main__":
    training_corpus = dp.generate_corpus(training_corpus_file, aqua, aqua2, human_judge)
    print("Hello World")
