import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import src.information_ordering as io
import src.data_preprocessing as dp
import src.content_selection as cs
import src.content_realization as cr


if __name__ == "__main__":
    print('Start...')

    # Paths and variables
    data_home = "."
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
        content_realization = cr.ContentRealization(solver="simple")  # use simple solver in cr
        content_realization.cr(sent_list, docset.idCode())


