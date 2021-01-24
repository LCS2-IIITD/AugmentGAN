# coding=utf-8
# ================== Useful packages imported =================
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import os
import random
import math
import multiprocessing 
import argparse
import tqdm
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import re
import nltk
import json
import pickle
import warnings
import warnings
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_recall_fscore_support
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

def code2text(file,wi_dict, iw_dict,write_file):
    '''Function to convert code/digits to text file'''
    f=open(file,"r")
    g=f.read().split('\n')
    a=""
    for i in g:
        for j in i.split(' '):
            try:
                a = a + iw_dict[str(j)] + " "
            except: 
                pass
        a=a+"\n"
    f=open(write_file,"w")
    f.write(a)

def calc_bleu(reference, hypothesis, weight):
    '''Function to calculate the BLEU Score value'''
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)

def get_reference_tokens(real_data):
    '''Getting the tokens for text and return the list of list of tokens for each line in text file'''
    reference = list()
    with open(real_data) as real_data:
        for text in real_data:
            text = nltk.word_tokenize(text)
            reference.append(text)
    return reference
 
def get_bleu_fast(test_data1,gram,real_data,sample_size):
    '''Calculate the BLEU Score fast'''
    reference = get_reference_tokens(real_data)
    random.shuffle(reference)
    reference = reference[0:sample_size]
    return get_bleu_parallel(test_data1,gram=gram,reference=reference)

def get_bleu_parallel(test_data1,gram,reference=None,):
    '''Getting the BLEU Score using parallel methodology'''
    ngram = gram
    weight = tuple((1. / ngram for _ in range(ngram)))
    pool = Pool(os.cpu_count())
    result = []
    with open(test_data1) as test_data:
        for hypothesis in test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            result.append(pool.apply_async(calc_bleu, args=(reference, hypothesis, weight)))
    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

def symmetric_diff(label1,label2):
    '''Calculate the symmetric difference for multilabel dataset'''
    d1=set(label2).symmetric_difference(set(label1))    
    return len(d1)

def cleanHtml(sentence):
    '''Clean the HTML tags if any '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence):
    '''Function to clean the word of any punctuation or special characters'''
    cleaned = re.sub(r'[!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    cleaned = cleaned.replace("unk"," ")
    cleaned = cleaned.replace("<unk>"," ")
    return cleaned

def keepAlpha(sentence):
    '''Function to keep the alphabets, full stop and question mark in text'''
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z.?]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def removeStopWords(sentence):
    '''Remove the Stop words'''
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def preprocess_test(test_text):
    '''Preprocess the text by removing the html tags, punctuations other than fullstop & question mark, and keeping the alphabets'''
    test=[]
    for i in test_text:
        if i=='' or i=="" or i==" " or i==' ':
            pass
        temp=i
        temp=temp.lower()
        temp=cleanHtml(temp)
        temp=cleanPunc(temp)
        temp=keepAlpha(temp)
        # temp=removeStopWords(temp)
        test.append(temp)
    return test
        
def hamming_loss(label1,label2,label_length):
    '''Function to calculate hamming loss'''
    q=label_length
    sum=0
    for i in range(len(label1)):    
        n=symmetric_diff(label1[i],label2[i])
        sum+=n/float(q)
    return sum/float(len(label1))

def ranking_loss_cal(y_true,y_score,y_pred,label_length):
    '''Function to calculate ranking loss'''
    sum=0
    for i in range(len(y_true)):
        count=0
        nr=list(set(y_pred[i]).difference(set(y_true[i])))
        r=list(set(y_pred[i]).intersection(set(y_true[i])))
        for j in range(len(nr)):
            for k in range(len(r)):
                if y_score[y_pred[i].index(nr[j])] > y_score[y_pred[i].index(r[k])]:
                    count+=1
        try:
            sum+=count/float(len(nr)*len(r))
        except:
            pass
    return sum/float(len(y_true))

def convert_label_test(pred,score,label_length):
    '''Making the prediction and corresponding score for multilabel test data set in scenario when it will predict no label (So, giving it another label of "label_length+1") to making in format for          running model)'''
    temp=[]
    temp2=[]
    for i in range(len(pred)):
        temp1=[]
        temp3=[]
        for j in range(len(pred[i])):
            if pred[i][j]==1:
                temp1.append(j)
                temp3.append(score[i][j])
        if temp1==[]:
            temp.append([label_length])
            temp2.append([1.0])
        else:

            temp.append(temp1)
            temp2.append(temp3)
    return temp,temp2

def convert_label_train(pred,label_length):
    '''Converting the predictions for train data in format for running model'''
    temp=[] 
    for i in range(len(pred)):
        temp1=[]
        for j in range(len(pred[i])):
            if pred[i][j]==1:
                temp1.append(j)
        if temp1==[]:
            temp.append([label_length])
        else:
            temp.append(temp1)
    return temp

def reduce_data(train_text):
    '''For reducing the dataset by removing less frequent words in dataset'''
    dict={}
    for i in range(len(train_text)):
        tr=train_text[i].split(' ')
        for j in range(len(tr)):
            try:
                dict[tr[j]]=dict[tr[j]]+1
            except:
                dict[tr[j]]=1
    count=0
    extras={}
    for i in dict.keys():
        if dict[i]<=2:
            extras[i]=0
    train_mine=[]
    for i in range(len(train_text)):
        tr=train_text[i].split(' ')
        temp=""

        for j in range(len(tr)):
            try:
                if extras[tr[j]]==0:
                    pass
            except:

                temp = temp + tr[j] +" "
        train_mine.append(temp)
    train_mine = np.array(train_mine)
    return train_mine

def subset_acc_main(label1,label2,count):
    '''Calculate the subset accuracy for muiltilabel dataset'''
    sum=0
    for i in range(len(label1)):
        
        tp=len(set(label1[i]).intersection(set(label2[i])))
        s1=symmetric_diff(label1[i],label2[i])
        tn=count-s1-tp
        try:
            sum+=(tp+tn)/float(count)
        except:
            pass
    return sum/float(len(label1))

def get_label_dist(label_dist,POSITIVE_FILE,POSITIVE_FILE1,train_label):
    '''Getting the data for particular label distribution'''
    train_label_main = pickle.load(open(train_label,"rb"))
    f=open(POSITIVE_FILE,"r")
    g=f.readlines()
    label = label_dist
    h=open(POSITIVE_FILE1+"_"+str(label_dist),"w")
    for i in range(len(train_label_main)):
        if train_label_main[i]==label:
            h.write(g[i])

def get_labels_with_more_samples(train_label,threshold):
    '''Getting the labels which has frequency greater than threshold'''
    train_label_main = pickle.load(open(train_label,"rb"))
    labels=[]
    for i in train_label_main:
        labels.append(i)
    dict={}
    for i in range(len(train_label_main)):
        try:
            dict[train_label_main[i]]=dict[train_label_main[i]] + 1
        except:
            dict[train_label_main[i]]=1
    real_labels=[]
    for i in dict:
        if dict[i]>threshold:
            real_labels.append(i)
    return real_labels


def scale(rewards, metric):
    '''Returning metric value in scale of reward'''
    mean1 = torch.mean(rewards)
    temp = metric
    while temp < mean1:
        temp = temp * 10
    temp = temp/10
    return temp
                              
def write_header_into_logfile(parameter_file,log_file,flag,BATCH_SIZE,TOTAL_BATCH,PRE_EPOCH_NUM,
                              PRE_EPOCH_NUM_DIS,g_steps,d_steps,epochs_dis,g_sequence_len,VOCAB_SIZE,data_loc):             
    '''Write the header description of experiments 
    Arguments :
    log_file : log file where the logs will get save
    flag : label distribution for which experiment is taken place
    BATCH_SIZE : Batch Size 
    TOTAL_BATCH : Number of Adversarial training steps
    PRE_EPOCH_NUM : Number of pre-training epochs for generator
    PRE_EPOCH_NUM_DIS : Number of pre-training epochs for discriminator
    g_steps : Number of steps in each training for generator
    d_steps : Number of steps in each training for discriminator
    epochs_dis : Number of steps in each pre-training for discriminator
    g_sequence_len: Sequence length
    VOCAB_SIZE : Vocab size of dataset
    data_loc : data location of dataset used
    '''
    log=open(log_file+str(flag),"w")
    log.write("Batch size : ")
    log.write(str(BATCH_SIZE))
    log.write("\n")
    log.write("Parameter File : ")
    log.write(str(parameter_file))
    log.write("\n")
    log.write("Adv Steps : ")
    log.write(str(TOTAL_BATCH))
    log.write("\n")
    log.write("PRE_EPOCH_NUM  : ")
    log.write(str(PRE_EPOCH_NUM))
    log.write("\n")
    
    log.write("PRE_EPOCH_NUM_DIS  : ")
    log.write(str(PRE_EPOCH_NUM_DIS))
    log.write("\n")
    log.write("g_steps  : ")
    log.write(str(g_steps))
    log.write("\n")
    log.write("d_steps  : ")
    log.write(str(d_steps))
    log.write("\n")
    log.write("epochs_dis  : ")
    log.write(str(epochs_dis))
    log.write("\n")
    log.write("Sequence length: ")
    log.write(str(g_sequence_len))
    log.write("\n")
    log.write("Vocab Size: ")
    log.write(str(VOCAB_SIZE))
    log.write("\n")
    log.write("Dataset used : ")
    log.write(data_loc)
    log.write("\n")
    log.write("Generating data ...")
    log.write("\n")
    log.close()

def sample_gen(model, encoder, batch_size, generated_num, output_file):
    samples = []
    g = read_file(POSITIVE_FILE)
    sample1 = random.sample(g,batch_size)
    sample1 = torch.from_numpy(np.array(sample1))
    zeros = torch.zeros((batch_size, 1)).type(torch.LongTensor)
    inputs = Variable(torch.cat([zeros, sample1], dim = 1)[:, :-1].contiguous())
    if cuda:
        inputs = inputs.cuda()
    h, c = init_hidden(batch_size,g_hidden_dim)
    pred,hidden,c = encoder.forward(inputs)
    return hidden
    
def generate_samples_conditional(model, encoder, batch_size, generated_num, output_file,g_sequence_len):
    '''Generate the samples with condition of noise'''
    samples = []
    g = read_file(POSITIVE_FILE)
    sample1 = random.sample(g,batch_size)
    sample1 = torch.from_numpy(np.array(sample1))
    zeros = torch.zeros((batch_size, 1)).type(torch.LongTensor)
    inputs = Variable(torch.cat([zeros, sample1], dim = 1)[:, :-1].contiguous())
    if cuda:
        inputs = inputs.cuda()
    h, c = init_hidden(batch_size,g_hidden_dim)
    pred,hidden,c = encoder.forward(inputs)
    
    for i in range(int(generated_num / batch_size)):
        sample = model.sample_conditional(batch_size,g_sequence_len, hidden).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)