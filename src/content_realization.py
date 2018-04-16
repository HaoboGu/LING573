#! /usr/bin/python
"""
    Content realization module, which converts information units to summary sentences.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.4.3
"""
import operator
from src.writer import write


class ContentRealization:
    """
    process method take summary content units(scu) as param and return summarized result
    """
    def __init__(self, scu_type="sentence", max_length=100):
        self.scu_type = scu_type
        self.max_length = max_length

    def get_summarizations(self, scu):
        """
        Generate summaries based on score of sentences
        :param scu: list of summary content units, which have score feature
        :return summary: list of summarized sentences
        """
        # Temporary take scu as sentence extracted, return sentence obj with highest salience score
        if self.scu_type == "sentence":
            # Sort sentence by score
            sorted_scu = sorted(scu, key=operator.attrgetter('score'), reverse=True)
            total_length = 0
            summary = []
            for item in sorted_scu:
                # Total length of summary should not larger than 100
                if total_length + len(item.sent) <= self.max_length:
                    summary.append(item.sent)
                    total_length += len(item.sent)
                else:
                    break

            return summary

    def cr(self, doc_set):
        """
        Process every doc in the docset and write the result to corresponding output file
        :type doc_set: class document
        """
        # Overwrite previous result in the first run
        doc = doc_set.documentCluster()[0]
        summary = self.get_summarizations(doc.sentences())
        write(summary, doc.topicID(), over_write=True)

        # Process other docs in docset
        for doc in doc_set.documentCluster()[1:]:

            summary = self.get_summarizations(doc.sentences())
            write(summary, doc.topicID(), over_write=False)

    def cr_for_single_doc(self, scu, topic_id):
        """
        Process a single doc and write the result to corresponding output file
        :param scu: list of summary content units, which have score feature
        :param topic_id: topic id of this doc
        """
        summary = self.get_summarizations(scu)
        write(summary, topic_id, over_write=False)

# Test script
# class SCU:
#     def __init__(self, sent, score):
#         self.sent = sent
#         self.score = score
#
#
# scus = []
# for i in range(0, 10):
#     sent = str(i)*(10+i)
#     score = i
#     scu = SCU(sent, score)
#     scus.append(scu)
#
# content_realization = ContentRealization()
# content_realization.cr_for_single_doc(scus, 'D1010A')





