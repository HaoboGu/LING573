#! /usr/bin/python
"""
    This is a writer module which writes required output to file.
    The output requirement can be found on Canvas.

    Author: Haobo Gu
    Data created: 04/14/2018
"""
import os


class Writer:
    """
    Take a list of strings as the input and write them to the output file.
    The output file should have one sentence per line.
    The output filename is generated using topic id.
    """
    def __init__(self, sentences, topic_id):
        """
        Initialization
        :param sentences: a list of output sentences
        :param topic_id: topic id for the sentences
        """
        self.sents = sentences
        self.topic_id = topic_id

    def write(self):
        """
        Write sentences to output file. If there is previous result, overwrite.
        """
        output_filename = os.path.join('outputs', self.generate_output_filename_from_topic_id())
        with open(output_filename, 'w') as output_file:
            # Write all sentences
            for sentence in self.sents:
                output_file.write(sentence+'\n')

    def generate_output_filename_from_topic_id(self):
        """
        Read topic_id, generate output filename.
        Topic_id should end with A or B, which is the docset_indicator.
        :return: output_filename, end with team number
        """
        docset_indicator = self.topic_id[-1]
        # Check the docset indicator
        if docset_indicator == "A" or "B":
            id_part1 = self.topic_id[:-1]
        else:
            print("docset_indicator should be either A and B")

        # our team number is 1
        return id_part1 + "-A.M.100." + docset_indicator + '.1'
