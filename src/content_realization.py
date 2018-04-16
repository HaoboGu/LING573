#! /usr/bin/python
"""
    Content realization module, which converts information units to summary sentences.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.4.3
"""
import operator


class ContentRealization:
    """
    process method take summary content units(scu) as param and return summarized result
    """
    def __init__(self, scu_type="sentence", max_length=100):
        self.scu_type = scu_type
        self.max_length = max_length

    def cr(self, scu):
        """
        :param scu: list of summary content units, which have salience scores feature
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


# Test script


# class SCU:
#     def __init__(self, sent, score):
#         self.sent = sent
#         self.score = score
#
# scus = []
# for i in range(0, 10):
#     sent = str(i)*(10+i)
#     score = i
#     scu = SCU(sent, score)
#     scus.append(scu)
#
# content_realization = ContentRealization()
# re = content_realization.cr(scus)
# print(re)




