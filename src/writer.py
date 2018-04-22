#! /usr/bin/python
"""
    This is a writer which contains methods write required output to file.
    The output requirement can be found on Canvas.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.4.3
"""
import os


def write(sentences, topic_id, over_write=True):
    """
    Write sentences to output file.
    :param sentences: summarization result
    :param topic_id: topic id for the summaries
    :param over_write: choose if overwrite the previous result
    :type sentences: list[str]
    :type topic_id: str
    :type over_write: bool
    """
    output_filename = generate_output_filename_from_topic_id(topic_id)
    if over_write:
        mode = 'w'
    else:
        mode = 'a'  # append to file
    with open(output_filename, mode) as output_file:
        # Write all sentences
        for sentence in sentences:
            output_file.write(sentence+'\n')


def generate_output_filename_from_topic_id(topic_id):
    """
    Generate output filename according to the requirement on Canvas
    :param topic_id: output filename contains topic_id information
    :type topic_id: str
    :return: output_filename, end with team number
    """
    id_part1 = topic_id[:-3]
    id_part2 = topic_id[-3]
    if 'outputs' in os.listdir('.'):
        path = 'outputs/D2'
    elif 'outputs' in os.listdir('..'):
        path = '../outputs/D2'
    else:
        print('cannot find output folder, store result in current folder')
        path = ''
    # our team number is 1
    return os.path.join(path, id_part1 + "-A.M.100." + id_part2 + '.1')


# Test script
# topic = "D0901H"
# topic2 = "D0808A"
# sents = ['1', '3', 'a', '4533333']
# write(sents, topic)
# write(sents, topic2)

