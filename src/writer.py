#! /usr/bin/python
"""
    This is a writer which contains methods write required output to file.
    The output requirement can be found on Canvas.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.4.3
"""
import os


def write(sentences, topic_id, output_folder_name='D3', over_write=True):
    """
    Write sentences to output file.
    :param sentences: summarization result
    :param topic_id: topic id for the summaries
    :param over_write: choose if overwrite the previous result
    :type sentences: list[str]
    :type topic_id: str
    :type output_folder_name: str
    :type over_write: boolean
    """
    output_filename = generate_output_filename_from_topic_id(topic_id, output_folder_name)
    if over_write:
        mode = 'w'
    else:
        mode = 'a'  # append to file
    with open(output_filename, mode) as output_file:
        # Write all sentences
        for sentence in sentences:
            output_file.write(sentence+'\n')


def generate_output_filename_from_topic_id(topic_id, output_folder_name):
    """
    Generate output filename according to topic id
    :type topic_id: str
    :type output_folder_name: str
    :return: output_filename, end with team number
    """
    id_part1 = topic_id[:-3]
    id_part2 = topic_id[-3]
    if 'outputs' in os.listdir('.'):
        path = 'outputs/' + output_folder_name
    elif 'outputs' in os.listdir('..'):
        path = '../outputs/' + output_folder_name
    else:
        print('cannot find output folder, store result in current folder')
        path = '' + output_folder_name
    # our team number is 1
    return os.path.join(path, id_part1 + "-A.M.100." + id_part2 + '.1')


# Test script
# topic = "D0901H"
# topic2 = "D0808A"
# sents = ['1', '3', 'a', '4533333']
# write(sents, topic)
# write(sents, topic2)

