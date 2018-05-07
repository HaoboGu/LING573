import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import src.information_ordering as io
import src.data_preprocessing as dp
import src.content_selection as cs
import src.content_realization as cr


if __name__ == "__main__":
    use_local_file = False
    # Paths and variables
    if use_local_file:
        # data_home = "/dropbox/17-18/573"
        data_home = ".."
        # training_corpus_file = data_home + "/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"
        training_corpus_file = data_home + "/Data/UpdateSumm09_demo_test_topics18.xml"
        aqua = data_home + "/AQUAINT"
        aqua2 = data_home + "/AQUAINT-2/data"
        human_judge = data_home + "/Data/models/training/2009"
    else:
        training_corpus_file = sys.argv[1]
        aqua = sys.argv[2]
        aqua2 = sys.argv[3]
        human_judge = sys.argv[4]
    print('Start...')

    comp_rate = 0.1  # The number of sentence selected
    converge_standard = 0.001  # Used to judge if the score is converging

    print("Reading Corpus...")
    training_corpus = dp.generate_corpus(training_corpus_file, aqua, aqua2, human_judge)
    docset_list = training_corpus.docsetList()
    docset_dic = {}

    for docset in docset_list:  # traverse through all document sets
        print("Processing docset", docset.idCode())
        important_sentences = cs.cs(docset, compression_rate=comp_rate)  # content selection
        sent_list = io.sort_sentence_list(important_sentences)  # sort important sentences
        content_realization = cr.ContentRealization(solver="improved_ilp", lambda1=0.5, lambda2=0.5,
                                                    output_folder_name='D3',
                                                    prune_pipe=['apposition', 'advcl', 'parenthesis'])
        content_realization.cr(sent_list, docset.idCode())
    print("Complete!")
