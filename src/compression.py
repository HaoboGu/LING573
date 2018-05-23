import xml.etree.ElementTree as ET
import os
import nltk
import pycrfsuite

stopwords = nltk.corpus.stopwords.words('english')
negation = set(['not', 'no', "n't", "nt"])

def check_stop(word):
    if word.lower() in stopwords:
        return True
    else:
        return False

def isnegation(word):
    if word in negation:
        return True
    else:
        return False


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isstop=%s' % check_stop(word),
        'word.isneg=%s' % isnegation(word),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.isstop=%s' % check_stop(word1),
            '-1:word.isneg=%s' % isnegation(word1),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.isstop=%s' % check_stop(word1),
            '-1:word.isneg=%s' % isnegation(word1),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

def label_sent(original, compressed):
    list_of_pos = list()
    ori_tokens = nltk.word_tokenize(original)
    comp_tokens = nltk.word_tokenize(compressed)
    ori_tags = nltk.pos_tag(ori_tokens)    

    flag = False
    counter = 0
    for idx, tok in enumerate(ori_tokens):
        if counter >= len(comp_tokens):
            list_of_pos.append((tok, ori_tags[idx][1], 'O'))
            continue
        if tok != comp_tokens[counter]:
            list_of_pos.append((tok, ori_tags[idx][1], 'O'))
            flag  = False
        elif tok == comp_tokens[counter] and flag == False:
            counter += 1
            list_of_pos.append((tok, ori_tags[idx][1], 'B'))
            flag = True
        else:
            counter += 1
            list_of_pos.append((tok, ori_tags[idx][1], 'I'))
    
    return list_of_pos

def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [label for (token, postag, label) in doc]

def read_training_file(dir):
    ori_to_comp = dict()
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        f = open(file_path, 'r', encoding='utf-8')
        contents = f.read()
        lines = contents.split('\n')
        original = ""
        for line in lines:
            if len(line.strip()) == 0 or line[0:6] == "<text>" or line[0:8] == "<doc id=" or line [0:7] == "</text>" or line[0:6] == "</doc>":
                continue
            else:
                split_once = line.split('>')
                sentence = (split_once[1].split('<'))[0].strip()
                if line[0:9] == "<original":
                    ori_to_comp[sentence] = ""
                    original = sentence
                elif line[0:11] == "<compressed":
                    ori_to_comp[original] = sentence
                else:
                    print(line)

    data_list = list()
    comp_list = list()
    ori_list = list()
    for sent in ori_to_comp:
        compressed = ori_to_comp[sent]
        pos = label_sent(sent, compressed)
        data_list.append(pos)
        comp_list.append(compressed)
        ori_list.append(sent)
    return data_list, ori_list, comp_list

def get_compressed(X, Y):
    addr = ""
    for idx, label in enumerate(Y):
        if label == "I" or label == "B":
            addr += (X[idx][1].split("=")[1]+" ")
    return addr.strip()

def compress_sent(sentence, tagger):
    sent_tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(sent_tokens)

    input_feat = extract_features(tags)
    y_pred = tagger.tag(input_feat)
    return get_compressed(input_feat, y_pred)

def create_a_tagger(training_file):
    data_list, ori_list, comp_list = read_training_file(training_file)
    X = [extract_features(doc) for doc in data_list]
    y = [get_labels(doc) for doc in data_list]
    
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 500,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
    })

    trainer.train('crf.model')
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    return tagger

if __name__ == "__main__":
    
    tagger = create_a_tagger("compression_corpus")
    print(compress_sent("He admired actors and praised those whose talents made his job easier .", tagger))
