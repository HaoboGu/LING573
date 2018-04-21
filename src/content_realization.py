#! /usr/bin/python
"""
    Content realization module, which converts information units to summary sentences.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.4.3
"""

from src.writer import write
import nltk
import numpy as np
from scipy.optimize import linprog


class ContentRealization:
    """
    process method take summary content units(scu) as param and return summarized result
    :param solver: indicate the default solver type
    """
    def __init__(self, solver="simple", max_length=100):
        self.solver = solver
        self.max_length = max_length

    def _get_summarizations(self, scu):
        """
        Generate summaries based on score of sentences
        :param scu: list of summary content units, which have been sorted
        :return summary: list of summarized sentences
        """
        total_length = 0
        summary = []
        sent_lens = get_lengths(scu)
        # Add sentences to summary
        for index, item in enumerate(scu):
            # Total length of summary should not larger than 100
            if total_length + sent_lens[index] <= self.max_length:
                summary.append(item.content())
                total_length += sent_lens[index]
        return summary

    def cr(self, scu, topic_id):
        """
        Content realization. Write the summary to output file.
        :param scu: list of sentence
        :param topic_id: topic id for this doc set
        """
        if self.solver == 'simple':
                self._simple_cr(scu, topic_id)
        elif self.solver == 'ilp':
            self._linear_prog(scu, topic_id)

    def _simple_cr(self, scu, topic_id):
        """
        Process a single docset and write the result to corresponding output file
        :param scu: list of summary content units, which have score feature
        :param topic_id: topic id of this doc
        """
        summary = self._get_summarizations(scu)
        write(summary, topic_id, over_write=True)

    def _linear_prog(self, scu, topic_id):
        """
        Integer linear programming solver. For details, please check:
        Extractive Multi-Document Summarization with Integer Linear Programming and Support Vector Regression
        :param scu: a list of sentence
        :param topic_id: topic id for this docset
        """
        bigram_dict, bigram_set = get_bigrams(scu)
        # Get coefficients for ILP
        target_coef, c1_coef, c2_coef, c3_coef, c1_value, c2_value, c3_value = \
            self._calculate_coef(scu, bigram_dict, bigram_set)

        bounds = tuple([(0, 1)]*len(c1_coef))  # bounds for variables
        coefs = np.concatenate((np.array([c1_coef]), c2_coef, c3_coef))  # coefs
        values = np.concatenate((np.array([c1_value]), c2_value, c3_value))  # values on right hand side
        sol = linprog(-target_coef, A_ub=coefs, b_ub=values, bounds=bounds, method="simplex")
        summary = []
        # Add sentences to summary
        for index, item in enumerate(scu):
            if abs(sol.x[index]-1) < 0.01:
                summary.append(item.content())

        # Write result
        write(summary, topic_id, over_write=True)

    def _calculate_coef(self, scu, bigram_dict, bigram_set, lambda1=0.5, lambda2=0.5):
        """
        Calculate coefficients for ILP.
        :param scu: a list of sentences
        :param bigram_dict: each sentence's bigrams
        :param bigram_set: set of all bigrams
        :param lambda1: weight for importance
        :param lambda2: weight for diversity
        :return: [target_coef, c1_coef, c2_coef, c3_coef, c1_value, c2_value, c3_value]
        """
        n_sentence = len(scu)  # number of sentences
        n_bigram = len(bigram_set)  # number of different bigrams
        bigram_list = list(bigram_set)

        # Get lengths for all sentences
        sent_lens = get_lengths(scu)
        min_sent_len = min(sent_lens)  # get number of words in shortest sentence

        # Calculate coefficients in target function
        # All coefs are divided to two parts: importance(imp) and diversity(div)
        target_imp_coef = np.zeros(n_sentence)
        target_div_coef = np.zeros(n_bigram)
        for i in range(n_sentence):
            target_imp_coef[i] = lambda1 * scu[i].score() * sent_lens[i] / (self.max_length/min_sent_len)
        for i in range(n_bigram):
            target_div_coef[i] = lambda2 / n_sentence
        target_coef = np.concatenate((target_imp_coef, target_div_coef), axis=0)

        # Calculate coefs in constraint 1
        c1_coef = np.concatenate((np.array(sent_lens), np.zeros(n_bigram)), axis=0)
        c1_coef = np.array(c1_coef)
        c1_value = np.array(self.max_length)

        # Calculate coefs in constraint 2, n_sentence constraints in total
        c2_coef = []
        for i in range(n_sentence):
            n_gram_i = len(bigram_dict[i])  # number of bigrams in sentence i
            c2_sent_coef = np.zeros(n_sentence)
            c2_sent_coef[i] = n_gram_i  # the coef of sentence i is n_gram_i
            c2_bi_coef = np.zeros(n_bigram)
            # If bigram appears in sentence i, set bigram's coef to -1
            for bigram in bigram_dict[i]:
                bigram_index = bigram_list.index(bigram)
                c2_bi_coef[bigram_index] = -1
            c2_coef.append(np.concatenate((c2_sent_coef, c2_bi_coef), axis=0))
        c2_value = np.zeros(n_sentence)  # value on right hand side
        c2_coef = np.array(c2_coef)

        # Calculate coefs in constraint 3, n_bigram constraints in total
        c3_coef = []
        c3_value = np.zeros(n_bigram)  # value on right hand side
        for j in range(n_bigram):
            cur_bigram = bigram_list[j]
            c3_sent_coef = np.zeros(n_sentence)
            c3_bi_coef = np.zeros(n_bigram)
            for i in range(n_sentence):
                # Current bigram appears in sentence i, set sentence's coef to -1
                if cur_bigram in bigram_dict[i]:
                    c3_sent_coef[i] = -1
            c3_bi_coef[j] = 1  # coef of current bigram is 1
            c3_coef.append(np.concatenate((c3_sent_coef, c3_bi_coef), axis=0))

        c3_coef = np.array(c3_coef)
        return target_coef, c1_coef, c2_coef, c3_coef, c1_value, c2_value, c3_value


# Helper functions
def get_lengths(scu):
    """
    Get lengths for all sentences
    :param scu: list of sentence
    :return: list of lengths
    """
    sent_lens = []  # list of lengths for sentences
    for sent in scu:
        sent_lens.append(len(nltk.word_tokenize(sent.content())))
    return sent_lens


def get_bigrams(scu):
    """
    Get all bigrams from a list of sorted scu
    :param scu: a list of sentences
    :return bigram_dict: each sentence's bigrams, key is the sentence's index
    :return bigram_set: a set of all bigrams
    """
    bigram_dict = {}
    bigram_set = set()
    for index, sent in enumerate(scu):
        bigrams = []
        word_seq = nltk.word_tokenize(sent.content())
        for word_index in range(0, len(word_seq)-1):  # for all bigrams
            bigrams.append((word_seq[word_index], word_seq[word_index+1]))
        bigram_dict[index] = bigrams  # key is the order of the sentence
        bigram_set = bigram_set.union(set(bigrams))
    return bigram_dict, bigram_set


# Test script
if __name__ == "__main__":
    import src.data_preprocessing as dp
    import src.content_selection as cs
    import src.information_ordering as io
    print('Start...')

    # Paths and variables
    data_home = ".."
    training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
    demo_training_corpus_file = data_home + "/Data/Documents/training/UpdateSumm09_demo_test_topics.xml"
    aqua = data_home + "/AQUAINT"
    aqua2 = data_home + "/AQUAINT-2/data"
    human_judge = data_home + "/Data/models/training/2009"
    comp_rate = 0.1  # The number of sentence selected
    converge_standard = 0.001  # Used to judge if the score is converging

    print("Reading Corpus...")
    training_corpus = dp.generate_corpus(demo_training_corpus_file, aqua, aqua2, human_judge)
    docset_list = training_corpus.docsetList()
    docset_dic = {}

    for docset in docset_list:  # traverse through all document sets
        print("Processing docset", docset.idCode())
        important_sentences = cs.cs(docset, compression_rate=comp_rate)  # content selection
        sent_list = io.sort_sentence_list(important_sentences)  # sort important sentences
        content_realization = ContentRealization(solver="ilp")  # use simple solver in cr
        content_realization.cr(sent_list, docset.idCode())




