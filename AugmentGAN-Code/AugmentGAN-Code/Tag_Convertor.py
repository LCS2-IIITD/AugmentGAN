import nltk
import numpy as np
import pandas as pd
import os
import sys

def tagging_sent_from_textfile(textfile,outfile):
    f = open(textfile,"r")
    g = f.read()
    g = g.split('\n')
    h = open(outfile,"w")
    for i in g:
        h = open(outfile,"a")
        text = nltk.word_tokenize(i)
        tagging = nltk.pos_tag(text)
        temp = []
        for i in tagging:
            temp.append(i[1])
        h.write(' '.join(temp))
        h.write('\n')
        h.close()
        
tagging_sent_from_textfile(sys.argv[1],sys.argv[2])
print("Converted................")
print("TAG File : ",sys.argv[2])