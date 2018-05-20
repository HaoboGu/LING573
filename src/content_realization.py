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
import pulp
import spacy
import re
import string
import src.data_preprocessing as dp


class ContentRealization:
    """
    process method take summary content units(scu) as param and return summarized result
    :param solver: indicate the default solver type
    :param max_length: the maximum summary length
    :param lambda1: weight for importance, used only with ilp solver
    :param lambda2: weight for diversity, used only with ilp solver
    """
    def __init__(self, solver="simple", max_length=100, lambda1=0.5, lambda2=0.5, output_folder_name='D3',
                 prune_pipe=('parenthesis', 'advcl', 'apposition')):
        self.solver = solver
        self.max_length = max_length
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.parenthesis_re = re.compile(r'(\[.+\])|(\(.+\))|(\{.+\})')  # match contents in parenthesis
        self.space_re = re.compile(r'\s+')  # match continuous spaces
        self.nlp = spacy.load('en')
        self.output_folder_name = output_folder_name
        self.prune_pipe = prune_pipe
        self.puncts = string.punctuation

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
        elif self.solver == 'improved_ilp':
            self._improved_ilp(scu, topic_id)
        elif self.solver == 'compression_ilp':
            self._compression_ilp(scu, topic_id)

    def prune_pipeline(self, scu, prune_pipe):
        """
        Prune sentences using method specified in prune_pipe
        :param scu: list of sentence
        :param prune_pipe: the order of pruning methods
        """
        for prune_type in prune_pipe:
            scu = self._prune_all(scu, prune_type)
        return scu

    def _prune_all(self, scu, prune_type):
        """
        Prune sentences by removing unnecessary constituents. Modify scu in place.
        :param scu: list of sentence
        :param prune_type: indicate the type of pruning
        """
        for index, item in enumerate(scu):
            sent = item.content()
            sent = self._prune_sentence(sent, prune_type)
            print('pruned sentence:', sent)
            # Re-calculate sentence length
            new_length = self._get_length(sent)
            # Update scu
            scu[index].set_content(sent)
            scu[index].set_length(new_length)

        return scu

    def _prune_sentence(self, sent, prune_type):
        """
        Apply prune compression methods on sentence and return pruned string
        :param sent: str
        :param prune_type: str
        :return: str
        """
        if prune_type == 'parenthesis':
            # Remove all contents in parenthesis
            sent = re.sub(self.parenthesis_re, '', sent).strip(' ')
            sent = re.sub(self.space_re, ' ', sent).strip(' ')
        elif prune_type == 'advcl':
            tokens = self.nlp(sent)
            if len(tokens) == 0:
                return ''
            if tokens[0].dep_ == 'advcl' or tokens[0].dep_ == 'prep':
                # If sentence starts with a short adv clause, remove this clause
                # Or if sentence starts with a short preposition phrase, remove it as well
                if ',' in sent[0:100]:
                    sent = sent[sent.find(',') + 1:].strip().capitalize()
        elif prune_type == 'apposition':
            tokens = self.nlp(sent)
            if len(tokens) == 0:
                return ''
            # Remove apposition
            for token in tokens:
                if token.dep_ == 'appos':
                    same_layer = [x for x in token.head.children]  # all elements in current layer
                    pos_in_layer = same_layer.index(token)  # index of apposition in current layer
                    if pos_in_layer - 1 >= 0 and pos_in_layer + 1 < len(same_layer) and \
                            same_layer[pos_in_layer - 1].dep_ == 'punct' and \
                            same_layer[pos_in_layer + 1].dep_ == 'punct':
                        appo_start = same_layer[pos_in_layer - 1].i
                        appo_end = same_layer[pos_in_layer + 1].i
                        sent = tokens[:appo_start].text + tokens[appo_end:].text
                        break  # prune only one apposition for each sentence
                    else:
                        # Get the subtree of apposition
                        subtree = [x for x in token.subtree]
                        if subtree[0].dep_ == 'punct':
                            appo_start = subtree[0].i
                        else:
                            appo_start = subtree[0].i - 1
                        if subtree[-1].dep_ == 'punct':
                            appo_end = subtree[-1].i
                        elif subtree[-1].i + 1 < len(tokens):
                            appo_end = subtree[-1].i + 1
                        else:
                            appo_end = subtree[-1].i
                        # If the apposition is surrounded by puncs, remove it
                        if tokens[appo_end].dep_ == 'punct' and tokens[appo_start].dep_ == 'punct':
                            sent = tokens[:appo_start].text + tokens[appo_end:].text
                            break  # prune only one apposition for each sentence
        return sent

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

    def _simple_cr(self, scu, topic_id):
        """
        Process a single docset and write the result to corresponding output file
        :param scu: list of summary content units, which have score feature
        :param topic_id: topic id of this doc
        """
        summary = self._get_summarizations(scu)
        write(summary, topic_id, output_folder_name=self.output_folder_name, over_write=True)

    def _linear_prog(self, scu, topic_id):
        """
        Integer linear programming solver. For details, please check:
        Extractive Multi-Document Summarization with Integer Linear Programming and Support Vector Regression
        :param scu: a list of sentence
        :param topic_id: topic id for this docset
        """
        bigram_dict, bigram_set = get_bigrams(scu)
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
            target_imp_coef[i] = self.lambda1 * scu[i].score() * sent_lens[i] / (self.max_length / min_sent_len)
        for i in range(n_bigram):
            target_div_coef[i] = self.lambda2 / n_sentence
        # target_coef = np.concatenate((target_imp_coef, target_div_coef), axis=0)

        # Calculate coefs in constraint 1
        # c1_coef = np.concatenate((np.array(sent_lens), np.zeros(n_bigram)), axis=0)

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
        c2_coef = np.array(c2_coef)

        # Calculate coefs in constraint 3, n_bigram constraints in total
        c3_coef = []
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

        ilp_model = pulp.LpProblem('content realization', pulp.LpMaximize)
        # Define variables
        sentences = pulp.LpVariable.dict("sentence", (i for i in range(n_sentence)),
                                         lowBound=0, upBound=1, cat=pulp.LpInteger)
        concepts = pulp.LpVariable.dict("bigram", (i for i in range(n_bigram)),
                                        lowBound=0, upBound=1, cat=pulp.LpInteger)
        # Add objective function
        ilp_model += pulp.lpSum([target_imp_coef[int(key)] * sentences[key] for key in sentences] +
                                [target_div_coef[int(key)] * concepts[key] for key in concepts])
        # Add length constraint
        ilp_model += pulp.lpSum([sent_lens[int(key)] * sentences[key] for key in sentences]) <= self.max_length
        # Add constraints 1
        for coefs in c2_coef:
            ilp_model += pulp.lpSum([coefs[key] * sentences[key] for key in sentences] +
                                    [coefs[key2 + n_sentence] * concepts[key2] for key2 in concepts]) <= 0
        # Add constraints 2
        for coefs in c3_coef:
            ilp_model += pulp.lpSum([coefs[key] * sentences[key] for key in sentences] +
                                    [coefs[key2 + n_sentence] * concepts[key2] for key2 in concepts]) <= 0

        # ilp_model.writeLP('ilp_model')  # write ilp model to file
        ilp_model.solve()
        indices = np.array([sentences[key].value() for key in sentences])
        summary = [sent.content() for sent in np.array(scu)[indices > 0.1]]

        # Write result
        write(summary, topic_id, output_folder_name=self.output_folder_name, over_write=True)

    def _improved_ilp(self, scu, topic_id):
        """
        An improved ILP algorithm for sentence realization. For details, please check:
        A Scalable Global Model for Summarization
        :param scu: list[Sentence]
        :param topic_id: topic id for this docset
        """
        # scu = self.prune_pipeline(scu, self.prune_pipe)
        bigram_dict, bigram_set = get_bigrams(scu)
        n_bigram = len(bigram_set)
        n_sent = len(scu)

        # Calculate weights in objective function
        # Count every bigram's occurrence
        bigram_freq = {}
        delta = 0.001 
        for index in bigram_dict:
            # For bigrams in each sentence[index]
            for bigram in bigram_dict[index]:
                # Bigrams in important sentences have higher weights
                if bigram not in bigram_freq:
                    bigram_freq[bigram] = 1 + delta * (n_sent - index)
                else:
                    bigram_freq[bigram] = bigram_freq[bigram] + 1 + delta * (n_sent - index)
        bigram_list = list(bigram_set)  # the order of bigram variables
        # Use frequency of bigram as its weight
        weight = []
        for i in range(n_bigram):
            weight.append(bigram_freq[bigram_list[i]])
        weight_s = np.zeros(n_sent)
        for i in range(n_sent):
            # Give a very small weight to every sentence
            weight_s[i] = delta * (n_sent - i)
        weight = np.concatenate((np.array(weight), weight_s), axis=0)  # add s_j after c_i

        # Calculate coefs
        # Variable: c_i, s_j
        n_constraint = 0
        coefs_1 = []  # constraint 1, i*j rows
        coefs_2 = []  # constraint 2, i rows
        for i in range(n_bigram):
            coefs_sum_s = np.zeros(n_sent)
            for j in range(n_sent):
                coefs_c = np.zeros(n_bigram)
                coefs_s = np.zeros(n_sent)
                if bigram_list[i] in bigram_dict[j]:
                    # c_i in s_j, s_j - c_i <= 0
                    coefs_c[i] = -1
                    coefs_s[j] = 1
                    coefs_sum_s[j] = -1
                    coefs_1.append(np.concatenate((coefs_c, coefs_s), axis=0))
                    n_constraint += 1
            coefs_sum_c = np.zeros(n_bigram)
            coefs_sum_c[i] = 1
            coefs_2.append(np.concatenate((coefs_sum_c, coefs_sum_s), axis=0))
            n_constraint += 1

        coefs_3_s = np.zeros(n_sent)  # constraint 3, length constraint, 1 row
        for i in range(n_sent):
            coefs_3_s[i] = scu[i].length()

        # Use pulp to solve ILP problem
        ilp_model = pulp.LpProblem('content realization', pulp.LpMaximize)
        # Define variables
        sentences = pulp.LpVariable.dict("sentence", (i for i in range(n_sent)),
                                         lowBound=0, upBound=1, cat=pulp.LpInteger)
        concepts = pulp.LpVariable.dict("concept", (i for i in range(n_bigram)),
                                        lowBound=0, upBound=1, cat=pulp.LpInteger)
        # Add objective function
        ilp_model += pulp.lpSum([weight[int(key)] * concepts[key] for key in concepts]), "Objective function"
        # Add length constraint
        ilp_model += pulp.lpSum([coefs_3_s[int(key)] * sentences[key] for key in sentences]) <= self.max_length
        # Add constraints 1
        for coefs in coefs_1:
            ilp_model += pulp.lpSum([coefs[key] * concepts[key] for key in concepts] +
                                    [coefs[key2+n_bigram] * sentences[key2] for key2 in sentences]) <= 0
        # Add constraints 2
        for coefs in coefs_2:
            ilp_model += pulp.lpSum([coefs[key] * concepts[key] for key in concepts] +
                                    [coefs[key2 + n_bigram] * sentences[key2] for key2 in sentences]) <= 0

        # ilp_model.writeLP('ilp_model')  # write ilp model to file
        ilp_model.solve(pulp.PULP_CBC_CMD())
        indices = np.array([sentences[key].value() for key in sentences])
        indices[indices==None] = 0
        summary = [sent.content() for sent in np.array(scu)[indices > 0.1]]
        # Write result
        write(summary, topic_id, output_folder_name=self.output_folder_name, over_write=True)

    def _generate_compressed_candidate(self, sent):
        """
        Generate compressed sentence candidates, including original sentence
        :param sent: sentence object
        :return: list[sentence]
        """
        candidate = [sent]
        for prune_type in self.prune_pipe:
            sent_content = sent.content()
            pruned_sent = self._prune_sentence(sent_content, prune_type)
            new_length = self._get_length(pruned_sent)
            candidate.append(dp.sentence(sent.idCode(), pruned_sent, sent.index(), sent.score(),
                                         new_length, sent.tokenDict(), sent.doctime()))
        return candidate

    def _compression_ilp(self, scu, topic_id):
        """
        Content realization by combining ILP and sentence compression
        :param scu: list[sentence]
        :param topic_id: str
        """
        # Get pruned sentences
        all_candidates = []
        for item in scu:
            all_candidates += self._generate_compressed_candidate(item)
        self._improved_ilp(all_candidates, topic_id)

    def _get_length(self, sent):
        """
        Get sentence length, without counting punctuations
        :param sent: str
        :return: int
        """
        n_puncs = 0
        word_seq = nltk.word_tokenize(sent)
        for word in word_seq:
            if word in self.puncts:
                n_puncs += 1
        new_length = len(word_seq) - n_puncs
        return new_length


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
    data_home = "."
    training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
    demo_training_corpus_file = data_home + "/Data/UpdateSumm09_demo_test_topics18.xml"
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
        content_realization = ContentRealization(solver="improved_ilp")  # use improved ILP solver in cr
        content_realization.cr(sent_list, docset.idCode())




