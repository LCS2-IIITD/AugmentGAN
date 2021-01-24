# coding=utf-8
# ================== Useful packages imported =================
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
from toolkit_functions import keepAlpha, removeStopWords, cleanPunc, cleanHtml
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def code_to_text(codes, dictionary):
    """For converting code (index) to text """
    
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    """ To tokenize and append into list"""
    tokenlized = list()
    with open(file,"r") as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    """making list of words (i.e, traverse all the list elements )"""
    tokens = sorted(tokens)
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(word_set)

def filter_func(word,threshold_extras):
    """" Filter function to get list of words i.e, vocab and extras(i.e, words whch are coming less frequently).
    Arguments :
    word : Sentences of words to be filter
    threshold_extras : Threshold on count of frequency of word to be considered as extras
    """
    word1=[]
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
    stemmer = SnowballStemmer("english")
    for i in range(len(word)):
        word1.append(cleanPunc(word[i]))
        # uncomment below for non-hindi
#         word1.append(keepAlpha(cleanPunc(cleanHtml(word[i]))))
    word2=[]
   
    for i in range(len(word1)):
        if word1[i]=='' or word1[i]=="":
            continue
        word2.append(word1[i])

    a={}
    for i in range(len(word2)):
        try:
            a[word2[i]]=a[word2[i]]+1
        except:
            a[word2[i]]=1
    wordset=[]
    extras={}
    store=list(a.keys())

    for i in range(len(a)):
        if (a[store[i]])>=threshold_extras:
            wordset.append(store[i])
        else:
            extras[store[i]]=1
    return wordset,extras
        
def get_dict(word_set):
    """To get the dictionary of words"""
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_to_code(tokens, dictionary, seq_len):
    """ Converting from text(word) to code  using mapping of both."""
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str = code_str[:-1]
        code_str += '\n'
    
    return code_str


def filtered_sentences(word):
    """To get the filtered sentence in list format"""
    word1=[]
    for i in range(len(word)):
#         if word[i] in stop_words:
#             continue
        # uncomment below for non-hindi
#         word1.append(keepAlpha(cleanPunc(cleanHtml(word[i]))))
        word1.append(cleanPunc(word[i]))
    word2=[]
    for i in range(len(word1)):
        if word1[i]==' ' or word1[i]=='' or word1[i]=="" or word1[i]==" ":
            continue
        word2.append(word1[i])
        
    return word2

def text_process_main(train_text_loc, extras,test_text_loc=None):
    """For processing the text and returning filtered sentences, word set and mapping of tokens with index""" 
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    
    data=train_tokens + test_tokens
    token=[]
    counter =0
    for i in range(len(data)):
        if counter%30000==0:
            print("Counter :",counter)
        counter =counter +1
        temp=[]
        
        a=filtered_sentences(data[i])
        
        for j in range(len(a)):
            try:
                a1= extras[a[j]]
                continue
            except:
                pass
            temp.append(a[j])
        token.append(temp)
                
    print("End")
    word_set = get_word_list(token)
    word_set=list(set(word_set))
    [word_index_dict, index_word_dict] = get_dict(word_set)

    return token, word_set, [word_index_dict, index_word_dict]

def text_process_extras(threshold_extras,train_text_loc, test_text_loc=None):
    '''Function to preprocess the text and get the extras (words whose occurence is less than user given threshold) in text '''
    train_tokens = get_tokenlized(train_text_loc)

    if test_text_loc is None:
        test_tokens = list()

    else:

        test_tokens = get_tokenlized(test_text_loc)
    
    word_set = get_word_list(train_tokens + test_tokens)
  
    word_set, extras=filter_func(word_set,threshold_extras)
  
    word_set=list(set(word_set))
    
    [word_index_dict, index_word_dict] = get_dict(word_set)
#     print(len(list(word_index_dict.keys())))
    print("Length of Extras :",len(extras))
    return len(word_index_dict) + 1, extras
