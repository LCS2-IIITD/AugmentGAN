#!/usr/bin/env python
# coding: utf-8

#Required packages
import pandas as pd
import numpy as np
from embedding import get_corpus
from nltk import pos_tag
import gensim
import re
import pickle
from nltk.corpus import wordnet as wn
# from textgenrnn import textgenrnn
import os
import nltk
import sys
import warnings
warnings.filterwarnings('ignore')

nltk.download('averaged_perceptron_tagger')
data=sys.argv[1]
itr=sys.argv[2]
fold=int(sys.argv[3])
label=int(sys.argv[4])
loc_save = sys.argv[5]
z=open(data,"r")
len1=len(z.readlines())-1
len2=int(len1/fold)

#Writing into particular file
def writing_into_file(aug_temp,target_path,y,x):
  
    g1=open(target_path,"a")
    for i in range(label):
        g1.write(y[i])
        g1.write(" ")
    g1.write("\t")
    g1.write(x)
    g1.write("\n")
    for i in range(1,len(aug_temp)):
        for j in range(label):
            g1.write(y[j])
            g1.write(" ")
        g1.write("\t")
        g1.write(aug_temp[i])
        g1.write("\n")

    
class Augment():

    def __init__(self,
                 method,
                 source_path,
                 target_path,
                 corpus_='none',
                 valid_tags=['NN'],
                 threshold=0.75,
                 x_col='tweet',
                 y_col='class'):
        """
        Constructor Arguments
        method (string):
            valid args:
                'postag': repalces all words of a given POS-tag in the sentence
                    with their most similar word vector from a large pre-trained
                    word embedding
                'threshold': Loads in a pre-trained word embedding and replaces
                    words in a sentence with the word vector of highest cosine
                    similarity
                'generative': trains a two-layer LSTM network to learn the word
                    representations of given class. The network then generates
                    samples of the class by initialising a random start word
                    and following the LSTM's predictions of the next word given
                    the previous sequence
        source_path (string): csv file that is meant to be augmented
        corpus_ (string): Word corpus that the similarity model should take in
            valid args: ['none', 'glove', 'fasttext', 'google']
        x_col (string): column name in csv from samples
        y_col (string): column name in csv for labels
        """
        f=[]
        #self.model = get_corpus(corpus_)
        self.model = pickle.load(open("google_bin","rb"))
        print('Loaded corpus: ', corpus_)
        self.x_col=x_col
        self.y_col=y_col
  
        self.df=pd.read_csv(source_path)
        self.augmented=pd.DataFrame(columns=[x_col, y_col])
        self.method=method
        self.valid_tags = valid_tags
        self.threshold_ = threshold
        g1=open(target_path,'a')
        # Go through each row in dataframe
        if method != 'generate':
            count=0
            f=[]
            count1=-1
            for idx, row in self.df.iterrows():
                count1+=1
                if count1>=(int(itr)-1)*len2 and count1<(int(itr))*len2:
                    try:
                        if count%200==0:
                            print("Counter :",count)
                        count+=1
    
                        x = self.preprocess(row[self.x_col])
                        pre=row[self.x_col]
                        y = str(int(row[self.y_col])).zfill(label)
       
                        if method =='postag':
                            aug_temp = self.postag(x)
                        if method =='threshold':
                        
                            aug_temp = self.threshold(x)
                            writing_into_file(aug_temp,target_path,y,pre)
                    except:
                        pass
                
                else:
                    pass

        else:
            self.generate('hate', 100)


    def preprocess(self, x):
        x = re.sub("[^a-zA-Z ]+", "", x)
        x = x.split()
        return x


    def postag(self, x):
        n = 0
        dict = {}
        tags = pos_tag(x)
        for idx, word in enumerate(x):
            if tags[idx][1] in self.valid_tags and word in self.model.wv.vocab:
                replacements = self.model.wv.most_similar(positive=word, topn=3)
                replacements = [elem[0] for elem in replacements]
                dict.update({word:replacements})
                n = len(replacements) if len(replacements) > n else n
        return self.create_augmented_samples(dict, n, x)

    def create_augmented_samples(self, dict, n, x):
        aug_tweets = [x]
        for i in range(n):
            single_augment = x[:]
            for idx, word in enumerate(single_augment):
                if word in dict.keys() and len(dict[word]) >= i+1:
                    single_augment[idx] = dict[word][i]
            single_augment = ' '.join(single_augment)
            aug_tweets.append(single_augment)

        return aug_tweets


    def threshold(self, x):
        dict = {}
        n = 0
        tags = pos_tag(x)
        for idx, word in enumerate(x):
            if word in self.model.wv.vocab:
                #get words with highest cosine similarity
                replacements = self.model.wv.most_similar(positive=word, topn=5)
                #keep only words that pass the threshold
                replacements = [replacements[i][0] for i in range(5) if replacements[i][1] > self.threshold_]
                #check for POS tag equality, dismiss if unequal
                replacements = [elem for elem in replacements if pos_tag([elem.lower()])[0][1] == tags[idx][1]]
                dict.update({word:replacements}) if len(replacements) > 0 else dict
                n = len(replacements) if len(replacements) > n else n
        return self.create_augmented_samples(dict, n, x)


#     def generate(self, class_, n_, to_file=False):
#         """
#         Takes in the name of the class and number of samples to be generated.
#         Trains a 2-layer LSTM network to generate new samples of given class
#         and writes them to the target path
#         Args:
#             class_(int): integer denoting the class number from dataframe/csv
#             n_(int): integer denoting the number of new samples to be created
#             toFile(bool): will write to self.target if set to True
#         """
#         df_class = self.df[self.df[y_col] == class_]
#         class_path = 'class_filter.csv'
#         df_class.to_csv(class_path) #write temporary csv file to train RNN instance
#         textgen = textgenrnn(name=self.method)
#         textgen.train_from_file(
#             file_path=class_path,
#             num_epochs=10,
#             batch_size=128,
#             new_model=True,
#             word_level=True
#         )

#         #generate new samples
#         if to_file:
#             textgen.generate_to_file(self.target, n=n_, temperature=1.0)
#         else: #will print to screen
#             textgen.generate(n_, temperature=1.0)

#         #clean up repo
#         os.remove(class_path)
#         os.remove(self.method+'_config.json')
#         os.remove(self.method+'vocab.json')
#         os.remove(self.method+'_weights.hdf5')


#Augmenting the data using threshold method
if __name__ == '__main__':
    Augment('threshold', data, str(loc_save)+'/augmented_data_nlp'+str(itr)+'.csv', 'google')



