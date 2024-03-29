import xml.etree.ElementTree as ET
import sys
import os
import re
import nltk
import string
import gzip
import spacy

class corpus:
    def __init__(self, docsetList, tokenDict):
        self._docsetList = docsetList
        self._tokenDict = tokenDict

    def docsetList(self):
        return self._docsetList

    def tokenDict(self):
        return self._tokenDict

class docSet:
    def __init__(self, idCode, documentCluster, humanSummary, tokenDict, topicID):
        self._idCode = idCode
        self._documentCluster = documentCluster
        self._humanSummary = humanSummary
        self._tokenDict = tokenDict
        self._topicID = topicID

    def idCode(self):
        return self._idCode
    
    def documentCluster(self):
        return self._documentCluster

    def humanSummary(self):
        return self._humanSummary

    def tokenDict(self):
        return self._tokenDict

    def topicID(self):
        return self._topicID

class document:
    def __init__(self, idCode, content, sentences, tokenDict, topicID, time):
        self._idCode = idCode
        self._content = content
        self._sentences = sentences
        self._topicID = topicID
        self._tokenDict = tokenDict
        self._time = time

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

    def time(self):
        return self._time

class sentence:
    def __init__(self, idCode, content, index, score, length, tokenDict, doctime):
        self._idCode = idCode
        self._content = content
        self._index = index
        self._score = score
        self._length = length
        self._tokenDict = tokenDict
        self._doctime = doctime

    def set_content(self, content):
        self._content = content

    def set_length(self, length):
        self._length = length

    def idCode(self):
        return self._idCode

    def content(self):
        return self._content
    
    def index(self):
        return self._index

    def score(self):
        return self._score

    def length(self):
        return self._length

    def tokenDict(self):
        return self._tokenDict

    def doctime(self):
        return self._doctime

def generate_corpus_from_xml(xml_file):
# This function generates a corpus skeleton from the xml file for training/devtest/evaltest
# This function reads in a xml file and returns a corpus type
    tree = ET.ElementTree(file=xml_file) # read in xml file as a parsing tree
    fullCorpus = corpus([],{}) # create a new corpus
    for topic in tree.iter(tag='topic'): # iter through all topics
        topic_id = topic.attrib["id"]
        for docSetA in topic.iter(tag='docsetA'): # iter through all docsetA
            docset_A_id = docSetA.attrib["id"]
            new_docset = docSet(docset_A_id, [], [], {}, topic_id) # create a new docset
            for doc in docSetA.iter(tag='doc'): # iter through all docs
                doc_id = doc.attrib["id"]
                time = doc_id[8:]
                new_doc = document(doc_id, "", [], {}, topic_id, time) # create a new doc
                new_docset._documentCluster.append(new_doc) # append the new doc to the docset's documentCluster member
            fullCorpus._docsetList.append(new_docset) # append the docset to the corpus' docsetList member
        """
        for docSetB in topic.iter(tag='docsetB'):
            docset_B_id = docSetB.attrib["id"]
            new_docset = docSet(docset_B_id, [], [], {}, topic_id)
            for doc in docSetB.iter(tag='doc'):
                doc_id = doc.attrib["id"]
                time = doc_id[8:]
                new_doc = document(doc_id, "", [], {}, topic_id, time)
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
        ori_file_path = os.path.join(dir1, first_dir_name.lower(), year, file_name) # update the file path for the original file
        with open(ori_file_path, 'rt') as f:
            with open(file_name+'_new', 'wt') as modified: # create a new file for reading
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
        if word in docset_dict: # if the word exists in the main dictionary
            docset_dict[word] += doc_dict[word] # add the count together
        else:
            docset_dict[word] = doc_dict[word] # set the count

def update_the_doc(doc, sentence_text, idx, doc_id, doc_time, nlp):
    index = idx
    sentence_text = sentence_text.replace('\n', ' ')
    sentence_text = re.sub('\s+', ' ', sentence_text)
    sentence_text = sentence_text.replace('``', '"')
    sentence_text = sentence_text.replace("''", '"')
    # sentences = nltk.tokenize.sent_tokenize(sentence_text)  # sent_tokenize the whole text
    sentences_list = nlp(sentence_text)  # Use spacy as sentence tokenizer
    sentences = [sent.text for sent in sentences_list.sents]
    for sent in sentences:
        word_tokens = nltk.tokenize.word_tokenize(sent)
        new_sent = sentence(doc_id, sent.strip(), index, 0, len(word_tokens), {}, doc_time)
        index += 1                            
        for word in word_tokens: # store the word count in the dictionary
            if word in string.punctuation:
                continue
            if word in doc._tokenDict:
                doc._tokenDict[hash(word)] += 1
            else:
                doc._tokenDict[hash(word)] = 1
            if word in new_sent._tokenDict:
                new_sent._tokenDict[hash(word)] += 1
            else:
                new_sent._tokenDict[hash(word)] = 1
        new_sent._length = len(new_sent._tokenDict)
        doc._sentences.append(new_sent) # append the sentence to the data structure
    return index


def fill_in_corpus_data(fullCorpus, dir1, dir2):
    # This function fill in the information of the corpus
    nlp = spacy.load('en', disable=["tagger", "parser", "ner"])
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    for docSet in fullCorpus._docsetList: # iterate through all document sets
        for doc in docSet._documentCluster: # iterate through all documents
            doc_id = doc._idCode
            doc_path, flag = generate_a_path(dir1, dir2, doc_id)
            doc_time = doc._time
            tree = ET.ElementTree(file=doc_path) # read the xml file as an elemen tree
            if flag == False:
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
                            update_the_doc(doc, text_all.text.strip(), 0, doc_id, doc_time, nlp)
                            break
                        else:
                            index = 0
                            for para in text_all.iter(tag='P'): # documents having <p> tag
                                update_idx = update_the_doc(doc, para.text.strip(), index, doc_id, doc_time, nlp)
                                index += update_idx
                            break
                os.remove(doc_path) # remove the file to release the space
            else:
                for sub_doc in tree.iter(tag='DOC'): # iterate through all sub-document in the doc
                    docno = sub_doc.attrib["id"]
                    if docno == None:
                        continue
                    if (docno != doc_id.strip()):
                        continue
                    else:
                        text_all = sub_doc.find('TEXT')
                        if text_all == None:
                            continue
                        para_all = text_all.find('P')
                        if para_all == None: # some documents does not have <p> tag, in that case, content is in <text>
                            update_the_doc(doc, text_all.text.strip(), 0, doc_id, doc_time, nlp)
                            break
                        else:
                            index = 0
                            for para in text_all.iter(tag='P'): # documents having <p> tag
                                update_idx = update_the_doc(doc, para.text.strip(), index, doc_id, doc_time, nlp)
                                index += update_idx
                            break
            update_dictionary(docSet._tokenDict, doc._tokenDict) # update the dictionary for the document set
        update_dictionary(fullCorpus._tokenDict, docSet._tokenDict) # update the dictionary for the corpus
    return fullCorpus

def fill_in_corpus_data_test(fullCorpus, dir):
    nlp = spacy.load('en', disable=["tagger", "parser", "ner"])
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    # This function fill in the information of the corpus
    for docSet in fullCorpus._docsetList: # iterate through all document sets
        for doc in docSet._documentCluster: # iterate through all documents
            doc_id = doc._idCode
            doc_path, flag = generate_a_path("", dir, doc_id)
            doc_path = doc_path.replace('.xml', '.gz')
                
            g = gzip.GzipFile(mode="rb", fileobj=open(doc_path, 'rb'))
            xml_file = open('try.xml', 'wb')
            xml_file.write(b'<DOCSTREAM>\n')
            xml_file.write(g.read())
            xml_file.write(b'</DOCSTREAM>')
            g.close()
            doc_time = doc._time
            xml_file.close()
            if doc_id == "XIN_ENG_20081118.0115":
                with open('try.xml', 'rt') as f:
                    with open('try_new.xml', 'wt') as modified: # create a new file for reading
                        for line in f:
                            modified.write(re.sub(r"<3", "", line)) # replace special tokens with space
                tree = ET.ElementTree(file='try_new.xml') # read the xml file as an elemen tree
            else:
                try:
                    tree = ET.ElementTree(file='try.xml') # read the xml file as an elemen tree
                except:
                    print(doc_id)
                    sys.exit(0)

            if flag == True:
                for sub_doc in tree.iter(tag='DOC'): # iterate through all sub-document in the doc
                    docno = sub_doc.attrib["id"]
                    if docno == None:
                        continue
                    if (docno != doc_id.strip()):
                        continue
                    else:
                        text_all = sub_doc.find('TEXT')
                        if text_all == None:
                            continue
                        para_all = text_all.find('P')
                        if para_all == None: # some documents does not have <p> tag, in that case, content is in <text>
                            update_the_doc(doc, text_all.text.strip(), 0, doc_id, doc_time, nlp)
                            break
                        else:
                            index = 0
                            for para in text_all.iter(tag='P'): # documents having <p> tag
                                update_idx = update_the_doc(doc, para.text.strip(), index, doc_id, doc_time, nlp)
                                index += update_idx
                            break
            update_dictionary(docSet._tokenDict, doc._tokenDict) # update the dictionary for the document set
        update_dictionary(fullCorpus._tokenDict, docSet._tokenDict) # update the dictionary for the corpus
    return fullCorpus

def read_human_judgements(fullCorpus, dir):
    # This function fills human judgements to the corpus
    for docSet in fullCorpus._docsetList:
        docset_id = docSet._idCode # id for the document set
        for file in os.listdir(dir):
            correct_name = docset_id[0:5]+'-'+docset_id[-1]+'.M.100.'+docset_id[5] # the name for the human judgement file
            if file[0:-2] == correct_name: 
                file_path = os.path.join(dir, file)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    docSet._humanSummary.append(str(data)) # load the human summary to the document set
    return fullCorpus

def set_index(corpus):
    for docset in corpus._docsetList:
        for doc in docset._documentCluster:
            num = 0
            for sent in doc._sentences:
                sent._index = num
                num += 1
    return corpus

def generate_corpus(corpus_file, aqua, aqua2, human_judge):
    # generate the corpus
    fullCorpus = generate_corpus_from_xml(corpus_file) # fill in all the essential information, create the corpus
    fullCorpus = read_human_judgements(fullCorpus, human_judge) # fill in the human judgements
    fullCorpus = fill_in_corpus_data(fullCorpus, aqua, aqua2) # fill in the data from two datasets
    fullCorpus = set_index(fullCorpus)
    return fullCorpus

def generate_test_corpus(corpus_file, test_data, human_judge):
    # generate the corpus
    fullCorpus = generate_corpus_from_xml(corpus_file) # fill in all the essential information, create the corpus
    fullCorpus = read_human_judgements(fullCorpus, human_judge) # fill in the human judgements
    fullCorpus = fill_in_corpus_data_test(fullCorpus, test_data) # fill in the data from two datasets
    fullCorpus = set_index(fullCorpus)
    return fullCorpus

if __name__ == "__main__":
    """
    training_corpus_file = sys.argv[1]
    aqua = sys.argv[2]
    aqua2 = sys.argv[3]
    human_judge = sys.argv[4]
    ifeval = sys.argv[5]
    """
    #fullCorpus = generate_corpus(training_corpus_file, aqua, aqua2, human_judge, ifeval)
    #fullCorpus = generate_test_corpus("dropbox/17-18/573/Data/Documents/evaltest/GuidedSumm11_test_topics.xml","dropbox/17-18/573/ENG-GW/data","dropbox/17-18/573/Data/models/evaltest")
    #fullCorpus = generate_test_corpus("evaltest.xml","eval","Data/models/evaltest")
    #fullCorpus = generate_corpus("test_train.xml", "LDC02T31", "LDC08T25/data", "Data/models/training/2009")
    #print(fullCorpus._docsetList[0]._documentCluster[0]._sentences[0]._content)

    #for sent in fullCorpus._docsetList[0]._documentCluster[0]._sentences:
    #    print(sent._index)
