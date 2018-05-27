#!/usr/bin/env python3

### imports
## modules
import src.data_preprocessing as dp
import src.content_selection as cs

## other packages
import operator
import nltk
import numpy as np
from scipy.spatial import distance

            
            
def calc_chro_exp(important_sentences):
    """
    ========================
    DESCRIPTION: this function takes an unsorted list of sentences and calculate the chronological expert
    ========================
    INPUT: important_sentences: list[sentence]
    ========================
    OUTPUT: chro_exp: chronological expert
    ========================
    """
    chro_exp = {}
    for i in important_sentences:
        if i not in chro_exp:
            chro_exp[i] = {}
        for j in important_sentences:
            #if j not in chro_exp[i] and i.index() != j.index():
            chro_exp[i][j] = 0
    for i in chro_exp:
        for j in chro_exp[i]:
            if i.doctime() < j.doctime():
                chro_exp[i][j] = 1
            elif i.idCode() == j.idCode() and i.index() < j.index():
                chro_exp[i][j] = 1
            elif i.doctime() == j.doctime() and i.idCode() == j.idCode():
                chro_exp[i][j] = 0.5
            else:
                chro_exp[i][j] = 0
    return(chro_exp)


def calc_cosine_sim(a,b):
    """
    ========================
    DESCRIPTION: this function calculates the cosine similarity between two sentences
    ========================
    INPUT: a,b,: two sentence objects
    ========================
    OUTPUT: cosine similarity score
    ========================
    """
    numerator = 0
    denominator = 0
    for i in a.tokenDict():
        if i in b.tokenDict():
            numerator+=a.tokenDict()[i]*b.tokenDict()[i]
    denominator = np.sqrt(len(a.tokenDict()))+np.sqrt(len(b.tokenDict()))
    return(numerator/denominator)

def get_doc_dic(docset):
    doc_cluster = docset.documentCluster()
    doc_dic = {}
    for i in doc_cluster:
        if i.idCode() not in doc_dic:
            doc_dic[i.idCode()] = i
    return(doc_dic)

def calc_topic_exp(u,v,Q):
    """
    ========================
    DESCRIPTION: this function calculates topic-closeness expert preference score
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: topic-closeness expert preference score
    ========================
    """
    topic_score_u = calc_topic_score(u,Q)
    topic_score_v = calc_topic_score(v,Q)
    if len(Q) == 0 or topic_score_u == topic_score_v:
        return(0.5)
    elif len(Q) != 0 and topic_score_u > topic_score_v:
        return(1.0)
    else:
        return(0)

def calc_topic_score(l,Q):
    """
    ========================
    DESCRIPTION: this function calculates topic-closeness similarity
    ========================
    INPUT: l: a sentence object
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: topic-closeness similarity
    ========================
    """
    if len(Q) == 0:
        return(0.0)
    else:
        max_sim_candidate = []
        for i in Q:
            max_sim_candidate.append(calc_cosine_sim(l,i))
        return(max(max_sim_candidate))


def calc_pre_exp(u,v,Q):
    """
    ========================
    DESCRIPTION: this function calculates precedence expert preference score
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: precedence expert preference score
    ========================
    """
    pre_score_u = calc_pre_score(u,v,Q)
    pre_score_v = calc_pre_score(v,u,Q)
    if len(Q) == 0 or pre_score_u == pre_score_v:
        return(0.5)
    elif len(Q) != 0 and pre_score_u > pre_score_v:
        return(1.0)
    else:
        return(0.0)

def calc_pre_score(u,v,Q,doc_dic):
    """
    ========================
    DESCRIPTION: this function calculates precedence similarity
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: precedence similarity
    ========================
    """
    if len(Q) == 0:
        return(0.0)
    else:
        sum_score = 0.0
        for q in Q:
            cur_doc = doc_dic[q.idCode()]
            pre_sent_sim = []
            for i in cur_doc.sentences():
                if i.index() < q.index():
                    pre_sent_sim.append(calc_cosine_sim(i,q))
            if len(pre_sent_sim) == 0:
                sum_score+=0
            else:
                sum_score+=max(pre_sent_sim)
        return(sum_score/len(Q)) 


def calc_succ_exp(u,v,Q):
    """
    ========================
    DESCRIPTION: this function calculates succession expert preference score
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: succession expert preference score
    ========================
    """
    succ_score_u = calc_succ_score(u,v,Q)
    succ_score_v = calc_succ_score(v,u,Q)
    if len(Q) == 0 or succ_score_u == succ_score_v:
        return(0.5)
    elif len(Q) != 0 and succ_score_u > succ_score_v:
        return(1.0)
    else:
        return(0.0)

def calc_succ_score(u,v,Q,doc_dic):
    """
    ========================
    DESCRIPTION: this function calculates succession similarity
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
    ========================
    OUTPUT: succession similarity
    ========================
    """
    if len(Q) == 0:
        return(0.0)
    else:
        sum_score = 0.0
        for q in Q:
            cur_doc = doc_dic[q.idCode()]
            succ_sent_sim = []
            for i in cur_doc.sentences():
                if i.index() < q.index():
                    succ_sent_sim.append(calc_cosine_sim(i,q))
            if len(succ_sent_sim) == 0:
                sum_score+=0
            else:
                sum_score+=max(succ_sent_sim)
        return(sum_score/len(Q)) 

def calc_total_pref(u,v,Q,chro_exp,doc_dic):
    """
    ========================
    DESCRIPTION: this function calculates the total preference
    ========================
    INPUT: u,v: two sentence objects
           Q: list of sentences that have been ordered
           chro_exp: chronological expert
    ========================
    OUTPUT: total preference
    ========================
    """
    chro_score = chro_exp[u][v]
    pre_score = calc_pre_exp(u,v,Q,doc_dic)
    succ_score = calc_succ_exp(u,v,Q,doc_dic)
    topic_score = calc_topic_exp(u,v,Q)
    return(0.327986*chro_score+0.016287*topic_score+0.196562*pre_score+0.444102*succ_score)
    


def sent_ordering(X,chro_exp):
    """
    ========================
    DESCRIPTION: this function takes an unsorted list of sentences and sort the sentences
    ========================
    INPUT: X: list[sentence]
    ========================
    OUTPUT: sorted_list: sorted sentence list
    ========================
    """
    V = X.copy()
    Q = []
    pi = {}
    rho = {}
    for i in V:
        pi[i] = 0
        rho[i] = 0
        
    for i in V:
        pi_i = 0
        for j in chro_exp[i]:
            pi_i+=calc_total_pref(i,j,Q,chro_exp,doc_dic)
        for j in chro_exp[i]:
            pi_i-=calc_total_pref(j,i,Q,chro_exp,doc_dic)
        pi[i] = pi_i
        
    while len(pi)!=0:
        t = max(pi, key=pi.get)
        rho[t] = len(V)
        V.remove(t)
        new_pi = {}
        for i in pi:
            if i != t:
                new_pi[i] = pi[i]
        pi = new_pi
        Q.append(t)
        for i in V:
            pi[i] = pi[i]+calc_total_pref(t,i,Q,chro_exp,doc_dic)-calc_total_pref(i,t,Q,chro_exp,doc_dic)
    sorted_list = sorted(rho, key=rho.get, reverse=True)
    return(sorted_list)


