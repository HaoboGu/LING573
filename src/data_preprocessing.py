import xml.etree.ElementTree as ET
import sys
import os
import re
import nltk

class corpus:
    def __init__(self, docsetList, tokenDict):
        self._docsetList = docsetList
        self._tokenDict = tokenDict

    def docsetList(self):
        return self._docsetList

    def tokenDict(self):
        return self._tokenDict

class docSet:
    def __init__(self, idCode, documentCluster, humanSummary, tokenDict):
        self._idCode = idCode
        self._documentCluster = documentCluster
        self._humanSummary = humanSummary
        self._tokenDict = tokenDict

    def idCode(self):
        return self._idCode
    
    def documentCluster(self):
        return self._documentCluster

    def humanSummary(self):
        return self._humanSummary

    def tokenDict(self):
        return self._tokenDict

class document:
    def __init__(self, idCode, content, sentences, tokenDict, topicID):
        self._idCode = idCode
        self._content = content
        self._sentences = sentences
        self._topicID = topicID
        self._tokenDict = tokenDict

    def idCode(self):
        return self._idCode

    def content(self):
        return self._content
    
    def sentences(self):
        return self._sentences

    def topicID(self):
        return self._topicID

    def tokenDict(self):
        return self._tokenDict

class sentence:
    def __init__(self, idCode, content, index, score):
        self._idCode = idCode
        self._content = content
        self._index = index
        self._score = score

    def idCode(self):
        return self._idCode

    def content(self):
        return self._content
    
    def index(self):
        return self._index

    def score(self):
        return self._score

def generate_corpus_from_xml(xml_file):
# This function generates a corpus skeleton from the xml file for training/devtest/evaltest
# This function reads in a xml file and returns a corpus type
    tree = ET.ElementTree(file=xml_file) # read in xml file as a parsing tree
    fullCorpus = corpus([],{}) # create a new corpus
    for topic in tree.iter(tag='topic'): # iter through all topics
        topic_id = topic.attrib["id"]
        for docSetA in topic.iter(tag='docsetA'): # iter through all docsetA
            docset_A_id = docSetA.attrib["id"]
            new_docset = docSet(docset_A_id, [], [], {}) # create a new docset
            for doc in docSetA.iter(tag='doc'): # iter through all docs
                doc_id = doc.attrib["id"]
                new_doc = document(doc_id, "", [], {}, topic_id) # create a new doc
                new_docset._documentCluster.append(new_doc) # append the new doc to the docset's documentCluster member
            fullCorpus._docsetList.append(new_docset) # append the docset to the corpus' docsetList member
        """
        for docSetB in topic.iter(tag='docsetB'):
            docset_B_id = docSetB.attrib["id"]
            new_docset = docSet(docset_B_id, [], [], {})
            for doc in docSetB.iter(tag='doc'):
                doc_id = doc.attrib["id"]
                new_doc = document(doc_id, "", [], {}, topic_id)
                new_docset._documentCluster.append(new_doc)
            fullCorpus._docsetList.append(new_docset)
        """
    return fullCorpus

def generate_a_path(dir1, dir2, doc_id):
# This function generates a path for the document
    if "_" in doc_id:
        doc_id_split = doc_id.lower().split('_')
        first_dir_name = doc_id_split[0]+'_'+doc_id_split[1] # find the first directory name
        file_id = doc_id_split[2].split('.') # find the file id
        file_name = first_dir_name+'_'+file_id[0][:-2]+'.xml' # generate the file name
        file_path = os.path.join(dir2, first_dir_name, file_name)
        flag = True
    else:
        first_dir_name = doc_id[0:3] # find the first directory
        year = doc_id[3:7] # find the year
        date = doc_id[7:11] # find the data
        if first_dir_name == 'APW':
            file_name = year+date+'_'+first_dir_name+'_ENG' # add the _ENG extenson
        elif first_dir_name == 'XIE':
            file_name = year+date+'_'+'XIN'+'_ENG' # add the _ENG extension and change XIN
        else:
            file_name = year+date+'_'+first_dir_name
        ori_file_path = os.path.join(dir1, first_dir_name.lower(), year, file_name)
        with open(ori_file_path, 'rt') as f:
            with open(file_name+'_new', 'wt') as modified:
                modified.write("<root>\n")
                for line in f:
                    modified.write(re.sub(r"&[A-Za-z0-9]+;", "", line)) # replace special tokens with space
                modified.write("</root>")
        file_path = file_name+'_new'
        flag = False
    return file_path, flag

def update_dictionary(docset_dict, doc_dict):
    # this function updates the docset_dict using entries from doc_dict
    # or the corpus dict from the docset_dict
    for word in doc_dict:
        if word in docset_dict:
            docset_dict[word] += doc_dict[word]
        else:
            docset_dict[word] = doc_dict[word]

def fill_in_corpus_data(fullCorpus, dir1, dir2):
    # This function fill in the information of the corpus
    for docSet in fullCorpus._docsetList: # iterate through all document sets
        for doc in docSet._documentCluster: # iterate through all documents
            doc_id = doc._idCode
            doc_path, flag = generate_a_path(dir1, dir2, doc_id)
            tree = ET.ElementTree(file=doc_path)
            for sub_doc in tree.iter(tag='DOC'): # iterate through all sub-document in the doc
                docno = sub_doc.find('DOCNO') # find the corresponding doc id
                if docno == None:
                    continue
                if (docno.text.strip() != doc_id.strip()):
                    continue
                else:  
                    body = sub_doc.find('BODY')    
                    text_all = body.find('TEXT')
                    if text_all == None:
                        continue
                    para_all = text_all.find('P')
                    if para_all == None: # some documents does not have <p> tag, in that case, content is in <text>
                        index = 0
                        sentences = nltk.tokenize.sent_tokenize(text_all.text) # sent_tokenize the whole text
                        for sent in sentences:
                            new_sent = sentence(doc_id, sent.strip(), index, 0)
                            index += 1
                            doc._sentences.append(new_sent) # append the sentence to the data structure
                            word_tokens = nltk.tokenize.word_tokenize(sent)
                            for word in word_tokens: # store the word count in the dictionary
                                if word in doc._tokenDict:
                                    doc._tokenDict[hash(word)] += 1
                                else:
                                    doc._tokenDict[hash(word)] = 1
                        break
                    else:
                        index = 0
                        for para in text_all.iter(tag='P'): # documents having <p> tag
                            new_sent = sentence(doc_id, para.text.strip(), index, 0)
                            index += 1
                            doc._sentences.append(new_sent)
                            word_tokens = nltk.tokenize.word_tokenize(para.text.strip())
                            for word in word_tokens: # store the word count in the dictionary
                                if word in doc._tokenDict:
                                    doc._tokenDict[hash(word)] += 1
                                else:
                                    doc._tokenDict[hash(word)] = 1
                        break
            if flag == False:
                os.remove(doc_path)
            update_dictionary(docSet._tokenDict, doc._tokenDict)
        update_dictionary(fullCorpus._tokenDict, docSet._tokenDict)
    return fullCorpus

def read_human_judgements(fullCorpus, dir):
    # This function fills human judgements to the corpus
    for docSet in fullCorpus._docsetList:
        docset_id = docSet._idCode
        for file in os.listdir(dir):
            correct_name = docset_id[0:5]+'-'+docset_id[-1]+'.M.100.'+docset_id[5]
            if file[0:-2] == correct_name:
                file_path = os.path.join(dir, file)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    docSet._humanSummary.append(data)
    return fullCorpus

def generate_corpus(corpus_file, aqua, aqua2, human_judge):
    # generate the corpus
    fullCorpus = generate_corpus_from_xml(corpus_file)
    fullCorpus = read_human_judgements(fullCorpus, human_judge)
    fullCorpus = fill_in_corpus_data(fullCorpus, aqua, aqua2)
    return fullCorpus

if __name__ == "__main__":
    training_corpus_file = sys.argv[1]
    aqua = sys.argv[2]
    aqua2 = sys.argv[3]
    human_judge = sys.argv[4]
    fullCorpus = generate_corpus(training_corpus_file, aqua, aqua2, human_judge)
