#! /usr/bin/python
"""
    Content realization module, which converts information units to summary sentences.

    Author: Haobo Gu
    Date created: 04/14/2018
    Python version: 3.6.2
"""
import operator


class ContentRealization:
    """
    process method take summary content units(scu) as param and return summarized result
    class SCU:
    def __init__(self, sent, score):
        self.sent = sent
        self.score = score
    """
    def __init__(self):
        self.scu_type = "sentence"

    def process(self, scu):
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
                if total_length + len(item.sent) <= 100:
                    summary.append(item.sent)
                    total_length += len(item.sent)
                else:
                    break
            return summary







