#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# ================== Useful packages imported =================
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
from loss import PGLoss
import torch.nn as nn
import re
import nltk
import copy
import json
import pickle
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from toolkit_functions import code2text, calc_bleu, get_bleu_fast, convert_label_test, convert_label_train, reduce_data, sample_gen
from toolkit_functions import symmetric_diff, preprocess_test, hamming_loss, ranking_loss_cal, generate_samples_conditional,code2text
from toolkit_functions import subset_acc_main, get_label_dist, get_labels_with_more_samples, scale, write_header_into_logfile
warnings.filterwarnings("ignore")


# In[ ]:





# In[2]:


def calculate_sequence(tokens):
    '''Function to return sequence length 
    Arguments :
    tokens : list of lists of token of each sentences
    '''
    d=[]
    for i in range(len(tokens)):
        d.append(len(tokens[i]))
    sequence_length=np.max(d)
    return sequence_length

def dataLoader(threshold_extras,POSITIVE_FILE,TRAIN_DATA_LOC=None):
    '''Function to load the data into positive file i.e, 
    file containing the original data with indices of words with padding with sequence length
    and to return the word <-> index map, sequence length and vocab size
    Arguments :
    threshold_extras : Threshold on count of frequency of word to be considered as extras
    TRAIN_DATA_LOC : data location where the original text file  
    '''
    from text_process import text_process_extras, text_process_main, text_to_code, get_tokenlized, get_word_list, get_dict
    vocab_size, extras = text_process_extras(threshold_extras,TRAIN_DATA_LOC)
    tokens, word_set, [word_index_dict, index_word_dict] = text_process_main(TRAIN_DATA_LOC, extras)
    sequence_length = calculate_sequence(tokens)
    
    # Writing the original file with indices of words to process for the model
    with open(POSITIVE_FILE, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, sequence_length))
    return word_index_dict, index_word_dict, sequence_length, vocab_size


def making_directories():
    '''Making directories for storing the augmented data and logs '''
    global dir_name_for_logs
    global AUG_LOC
    try:
        os.mkdir(main_folder_save)
    except:
        pass
    try:
        os.mkdir(dir_name_for_logs)
    except:
        pass
    try:
        os.mkdir(AUG_LOC)
    except:
        pass  

def eval_epoch_bleu(EVAL_FILE, data_loc,flag,write_file,wi_dict,iw_dict):
    '''Evaluate the generated samples with original or real data
    return the similarity measure i.e, Bleu score
    '''
    f=open(data_loc)
    length=len(f.read().split('\n'))
    code2text(EVAL_FILE,wi_dict, iw_dict,write_file+str(flag))
    return get_bleu_fast(write_file+str(flag),2,data_loc,int(length/50))

    
    
def generate_samples(model, batch_size, generated_num, output_file,g_sequence_len):
    '''Generate the augmented samples as per given '''
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
            
            
class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
#         loss =  -torch.sum(loss)/len(loss)
        loss =  -torch.sum(loss)
        return loss

def train_epoch(model, data_iter, criterion, optimizer):
    '''For training at each epoch for given model '''
    total_loss = 0.
    total_words = 0.
    
    for (data, target) in data_iter:#tqdm(
        data = Variable(data)
        target = Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
#         print("target : ",target)
#         print("data: ",data)
        pred,_,_ = model.forward(data)
        
        
        loss = criterion(pred, target)
        
#         print("loss : ",loss)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)

def eval_epoch_loss(model, data_iter, criterion):
    '''For evaluating at each epoch'''
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:
           
            data = Variable(data)
            target = Variable(target)
            if cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred,_,_ = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)


# In[ ]:





# In[ ]:





# In[3]:



# ================== Parameter Definition =================
'''Commandline  Argument will be json file containing the keys as :  
    Parameter 1 : data location for train data 
    Parameter 2 : Number of pre-training epochs for generator
    Parameter 3 : Number of Adversarial training steps
    Parameter 4 : Number of pre-training epochs for discriminator
    Parameter 5 : Batch Size
    Parameter 6 : Number of steps in each training for discriminator
    Parameter 7 : Number of steps in each training for generator
    Parameter 8 : label distribution for which experiment is taken place
    Parameter 9 : For finetuning for last few epoch i.e, after seed number ;using particular label distribution
    Parameter 10: location to save the augmented data
    Parameter 11: Number of samples to generate
    Parameter 12: Number of steps in each pre-training for discriminator
    Parameter 13: Label length i.e, for single label data set ; Use 1 ; Otherwise use number of labels it has
    Parameter 14: Type of reward i.e, either to 0 : for subtract or divide and 1 : for  to multiply or add.
    Parameter 15: Maximum value of reward i.e, in case of bleu, it is 1.
    Parameter 16: Minimum loss generator can have . Used for condition on generator for epochs
    Parameter 17: suffix to directory made for storing the augmented data for different experiments
    Parameter 18: Threshold on frequency of words to be considered as extras for minimizing the vocab size i.e, for toxic  = 10;imdb = 20;sst = 0.
    Parameter 19: Location of training label pickled file
    Parameter 20: Flow in adversarial training i.e, 1: G--->D or 0:  D--->G. 
    Parameter 21: plugged-in flag i.e, # bleu, subset_acc, ham_loss, rank_loss
    Parameter 22: plugin operation i.e,  # add, multiply
    Parameter 23: To give condition of hidden embedding or not. i.e, # 1 : conditional and 0 : Non-conditional
    Parameter 24 : Epsilon value for convergence criteria
    Parameter 25 : Experiment name (Main folder where each and everything will get store)   
    '''

SEED = 88
cuda = True
output_file = open(sys.argv[1]).read()
parameter_json = json.loads(output_file)
main_folder_save = parameter_json['main_folder_save']
TAG_DATA_LOC = parameter_json['TAG_DATA_LOC']
TRAIN_DATA_LOC = parameter_json['TRAIN_DATA_LOC']
PRE_EPOCH_NUM_GEN = int(parameter_json['PRE_EPOCH_NUM_GEN'])
PRE_EPOCH_NUM_GEN_TAG = int(parameter_json['PRE_EPOCH_NUM_GEN_TAG'])
ADV_STEPS_FINE =  int(parameter_json['ADV_STEPS_FINE'])
ADV_STEPS = int(parameter_json['ADV_STEPS'])
ADV_STEPS_TAG = int(parameter_json['ADV_STEPS_TAG'])
PRE_EPOCH_NUM_DIS = int(parameter_json['PRE_EPOCH_NUM_DIS'])
PRE_EPOCH_NUM_DIS_TAG = int(parameter_json['PRE_EPOCH_NUM_DIS_TAG'])
BATCH_SIZE = int(parameter_json['BATCH_SIZE'])
D_STEPS=int(parameter_json['D_STEPS'])
G_STEPS=int(parameter_json['G_STEPS'])
SEED_ON_LABEL_EPOCH = int(parameter_json['SEED_ON_LABEL_EPOCH'])
DIRECTORY_LABEL = None
GENERATED_NUM = int(parameter_json['GENERATED_NUM'])
EPOCH_IN_DIS = int(parameter_json['EPOCH_IN_DIS'])
label_length = int(parameter_json['label_length'])
rew_flag = int(parameter_json['rew_flag'])
max_val = int(parameter_json['max_val'])
min_loss = float(parameter_json['min_loss'])
experiment = parameter_json['experiment']
POSITIVE_FILE =main_folder_save+"/" + 'real_'+str(parameter_json['POSITIVE_FILE'])+'.data' + experiment
POSITIVE_FILE_TAG =main_folder_save+"/" + 'realtag_'+str(parameter_json['POSITIVE_FILE'])+'.data' + experiment
AUG_LOC = main_folder_save + "/" + parameter_json['AUG_LOC'] +  experiment
POSITIVE_FILE1= main_folder_save + "/" + 'label_local.data' +experiment
POSITIVE_FILE1_TAG= main_folder_save + "/" + 'label_local_tag.data' +experiment
NEGATIVE_FILE = main_folder_save + "/" + 'gene.data' + experiment
NEGATIVE_FILE_TAG = main_folder_save + "/" + 'gene_tag.data' + experiment
EVAL_FILE = main_folder_save + "/" +'eval.data' + experiment
EVAL_FILE_TAG = main_folder_save + "/" +'eval_tag.data' + experiment
dir_name_for_logs = main_folder_save + "/" + "parameter" + experiment
threshold_extras = int(parameter_json['threshold_extras'])
train_label = parameter_json['train_label']
gen_dis_flag = int(parameter_json['gen_dis_flag'])  # 0 means discriminator -> generator
                                                                            # else, means generator -> discriminator
plugin_flag = parameter_json['plugin_flag'] # bleu, subset_acc, ham_loss, rank_loss
plugin_operation = parameter_json['plugin_operation'] # add, multiply
conditional_flag = int(parameter_json['conditional_flag']) # 1 : conditional and 0 : Non-conditional
epsilon = float(parameter_json['epsilon'])
making_directories()
log_file =  dir_name_for_logs+"/log.txt"
test_file_loc = "test_file.txt"
test_file_loc_tag = "test_file_tag.txt"
write_file= main_folder_save +"/" + "write_file.txt"
write_file_tag= main_folder_save +"/" + "write_file_tag.txt"
temp_file = main_folder_save +"/" + "temp_generated.txt"
temp_file_tag = main_folder_save +"/" + "temp_generated_tag.txt"
# Generator Parameters for data
g_emb_dim = 300
g_hidden_dim = 32

# Discriminator Parameters
d_emb_dim = 300
d_dropout = 0.75
d_num_class = 2
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
# d_num_filters_tag = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10]


# In[ ]:





# In[4]:


# Getting the word->index and index->word mappings, vocab size and sequence length of data
wi_dict, iw_dict, g_sequence_len, VOCAB_SIZE = dataLoader(threshold_extras,POSITIVE_FILE,TRAIN_DATA_LOC)
print("Sequence Length : ",g_sequence_len,"\n","Vocab Size : ",VOCAB_SIZE)
print("Step 2 : Writing the code data correponding to text into real.data")


# In[ ]:





# In[ ]:





# In[5]:


# Getting the word->index and index->word mappings, vocab size and sequence length of tag_data
wi_dict_tag, iw_dict_tag, g_sequence_len_tag, VOCAB_SIZE_TAG = dataLoader(threshold_extras,POSITIVE_FILE_TAG,TAG_DATA_LOC)
print("Sequence Length : ",g_sequence_len_tag,"\n","Vocab Size : ",VOCAB_SIZE_TAG)
print("Step 2 : Writing the code data correponding to text into real_tag.data")


# In[ ]:





# In[6]:


# # To add <eos> as last token
# wi_dict["<eos>"] = str(VOCAB_SIZE -1)
# iw_dict[str(VOCAB_SIZE -1)] = "<eos>"
# wi_dict_tag["<eos>"] = str(VOCAB_SIZE_TAG -1)
# iw_dict_tag[str(VOCAB_SIZE_TAG -1)] = "<eos>"


# In[7]:


# Made copy into the new variable for tag.
# g_sequence_len_tag = g_sequence_len


# In[8]:



'''Main function to run the GAN architecture 
Arguments :
flag : label distribution to use for the GAN to run on particular label distribution 
'''
flag = int(parameter_json['POSITIVE_FILE'])
np.random.seed(SEED)
# Need to append the other details of TAG related stuff in log file
write_header_into_logfile(output_file,log_file,flag,BATCH_SIZE,ADV_STEPS,PRE_EPOCH_NUM_GEN,
                          PRE_EPOCH_NUM_DIS,G_STEPS,D_STEPS,EPOCH_IN_DIS,g_sequence_len,VOCAB_SIZE,TRAIN_DATA_LOC)

# Constructing generator and discriminator
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, cuda)
encoder = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, cuda)
discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
generator_tag = Generator(VOCAB_SIZE_TAG, g_emb_dim, g_hidden_dim, cuda)
encoder_tag = Generator(VOCAB_SIZE_TAG, g_emb_dim, g_hidden_dim, cuda)
discriminator_tag = Discriminator(d_num_class, VOCAB_SIZE_TAG, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    encoder = encoder.cuda()
    encoder_tag = encoder_tag.cuda()
    generator_tag = generator_tag.cuda()
    discriminator_tag = discriminator_tag.cuda()
    
DIRECTORY_LABEL = str(flag)
os.chdir(dir_name_for_logs)
try:
    os.mkdir(DIRECTORY_LABEL)
except:
    pass
os.chdir('../..')

get_label_dist(flag,POSITIVE_FILE,POSITIVE_FILE1,train_label)
get_label_dist(flag,POSITIVE_FILE_TAG,POSITIVE_FILE1_TAG,train_label)



# In[9]:


# Load data from original file for first iteration
# Load the data for particular label distribution
gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
gen_data_iter_label = GenDataIter(POSITIVE_FILE1 + "_" + str(flag), BATCH_SIZE)
gen_data_iter_tag = GenDataIter(POSITIVE_FILE_TAG, BATCH_SIZE)
gen_data_iter_label_tag = GenDataIter(POSITIVE_FILE1_TAG + "_" + str(flag), BATCH_SIZE)
# Pretrain Generator using NLL Loss and Adam optimizer
gen_criterion = nn.NLLLoss(reduction='sum')
lr = 0.001
gen_optimizer = optim.Adam(generator.parameters(),lr = lr)
gen_criterion_tag = nn.NLLLoss(reduction='sum')
gen_optimizer_tag = optim.Adam(generator_tag.parameters(),lr =lr)
if cuda:
    gen_criterion = gen_criterion.cuda()
    gen_criterion_tag = gen_criterion_tag.cuda()

log=open(log_file+str(flag),"a")
log.write("Pretrain with MLE ...")
log.write("\n")
log.close()
print('Pretrain with MLE ...')


# In[10]:


def generator_training(PRE_EPOCH_NUM_GEN,SEED_ON_LABEL_EPOCH,generator,gen_data_iter_label,gen_criterion,gen_optimizer,
                       log_file,flag,gen_data_iter,dir_name_for_logs,DIRECTORY_LABEL,BATCH_SIZE,GENERATED_NUM,EVAL_FILE,TRAIN_DATA_LOC,write_file,
                      wi_dict,iw_dict,g_sequence_len):
    '''For training the generator for some epochs
    
    '''


    gen_loss=[]
    gen_bleu=[]
    end = 10000 # just a big number
    for epoch in range(PRE_EPOCH_NUM_GEN):
        # For finetuning for last few epoch i.e, after seed number ;using particular label distribution
        if epoch >= SEED_ON_LABEL_EPOCH:
            # Pretrained on particular label distribution
            loss = train_epoch(generator, gen_data_iter_label, gen_criterion, gen_optimizer)
            end = loss
            log=open(log_file+str(flag),"a")
            log.write('Epoch [%d] Model Loss: %f'% (epoch, loss))
            log.write("\n")    
        else:
            # For all dataset (Pre-trained on complete dataset ) 
            loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
            log=open(log_file+str(flag),"a")
            log.write('Epoch [%d] Model Loss: %f'% (epoch, loss))
            log.write("\n")

        if epoch%2==0:
            torch.save(generator,  "./"+dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_pregen"+str(epoch))
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)


        gen_loss.append(loss)
    
        if abs(loss-end) < epsilon and loss < min_loss:
                break
        end = loss

    return generator,gen_loss
#Initialising the encoder condition for generator


# In[17]:


def discriminator_training(PRE_EPOCH_NUM_DIS,conditional_flag,generator,encoder,BATCH_SIZE,GENERATED_NUM, NEGATIVE_FILE,POSITIVE_FILE,
                          EPOCH_IN_DIS,discriminator, dis_criterion, dis_optimizer,dir_name_for_logs,DIRECTORY_LABEL,g_sequence_len):
    dis_loss = []
    for epoch in range(PRE_EPOCH_NUM_DIS):
        if conditional_flag == 1:
            generate_samples_conditional(generator, encoder, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE,g_sequence_len)
        else:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE,g_sequence_len)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        losses = 0
        for count in range(EPOCH_IN_DIS):
        
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            losses+= loss
         
            if epoch%5==0:
                torch.save(discriminator,  "./"+ dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_predis"+str(epoch)+str(count))
            log=open(log_file+str(flag),"a")
            log.write('Epoch [%d], loss: %f' % (epoch, loss))
            log.write("\n")
            log.close()
        dis_loss.append(losses/EPOCH_IN_DIS)
            
    return discriminator,dis_loss


# In[ ]:





# In[12]:


def convert_textcode2tagcode(sample,wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,VOCAB_SIZE,VOCAB_SIZE_TAG):
    text = []
    for i in range(len(sample)):
        try:
            text.append(iw_dict[str(sample[i])])
        except:
            text.append(" ")
   
    tagging = nltk.pos_tag(text)
    temp = []
    for i in tagging:
        temp.append(i[1])
    text = []
    temp1 = [i.lower() for i in temp]
    for i in range(len(temp1)):
        try:
            text.append(int(wi_dict_tag[temp1[i]]))
        except:
            text.append(VOCAB_SIZE_TAG-1)
    return np.array(text)


def text_code2tag_code(samples,wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,VOCAB_SIZE,VOCAB_SIZE_TAG):
    
    samples = samples.cpu().numpy()
    output = []
    for i in samples:
        output.append(convert_textcode2tagcode(i,wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,VOCAB_SIZE,VOCAB_SIZE_TAG))
    output = torch.from_numpy(np.array(output))
    return output.cuda()


# In[13]:


def adversarial_training(generator,encoder,discriminator,generator_tag,encoder_tag,discriminator_tag,TRAIN_DATA_LOC,BATCH_SIZE,
                               GENERATED_NUM,NEGATIVE_FILE,g_sequence_len,G_STEPS,D_STEPS,ADV_STEPS,EVAL_FILE,wi_dict,iw_dict,wi_dict_tag,
                               iw_dict_tag,AUG_LOC,test_file_loc,flag,write_file,max_val,log_file,dis_criterion,dis_optimizer,gen_gan_optm,conditional_flag,total_batch,
                               gen_dis_flag,rollout,POSITIVE_FILE,VOCAB_SIZE,VOCAB_SIZE_TAG,mixed_flag):
    # Train the generator for one step
    # Discriminator ---> Generator
    # Train the generator for one step
    if gen_dis_flag==0:

        for count in range(D_STEPS):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE,g_sequence_len)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for count1 in range(EPOCH_IN_DIS):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
                log=open(log_file+str(flag),"a")
                log.write('Loss : %f' % (loss))
                log.write("\n")
                log.close()
                if count%2==0:
                    torch.save(discriminator,  "./"+ dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_dis"+str(count)+str(count1))
                    torch.save(generator,  "./"+ dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_gen"+str(count)+str(count1))

        for it in range(G_STEPS):
            if conditional_flag==1:
                hidden = sample_gen(generator, encoder, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
                samples = generator.sample_conditional(BATCH_SIZE, g_sequence_len, hidden)
            else:
                samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
#             print(samples)
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            ####
            # Used for mixed adversarial training
            if mixed_flag ==0:
                rewards = rollout.get_reward(samples, 16, discriminator)
            else:
                samples_tag = text_code2tag_code(samples,wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,VOCAB_SIZE,VOCAB_SIZE_TAG)
                # calculate the reward
                rewards_tag = rollout_tag.get_reward(samples_tag, 16, discriminator_tag)
                rewards_text = rollout.get_reward(samples, 16, discriminator)
                rewards = rewards_tag + rewards_text
            ####
        
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if conditional_flag == 1:
                generate_samples_conditional(generator, encoder, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            else:
                generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            code2text(EVAL_FILE,wi_dict, iw_dict,str(AUG_LOC)+"/"+test_file_loc+str(flag))
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)

            if plugin_flag == "bleu":
                bleu = eval_epoch_bleu(EVAL_FILE, TRAIN_DATA_LOC,flag,write_file,wi_dict,iw_dict)
                bleu = scale(rewards,bleu)
                if plugin_operation == "add":
                    if rew_flag == 0:
                        rewards = rewards - bleu
                    else:
                        rewards = rewards + bleu
                else:
                    if rew_flag == 0:
                        rewards = rewards * (max_val - bleu)
                    else:
                        rewards = rewards * bleu                   
            else:
                plugin = plugin_toolkit(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE, temp_file, label_length,plugin_flag)
                plugin = scale(rewards,plugin)
                if plugin_operation == "add":
                    if rew_flag == 0:
                        rewards = rewards - plugin
                    else:
                        rewards = rewards + plugin
                else:
                    if rew_flag == 0:
                        rewards = rewards * (max_val - plugin)
                    else:
                        rewards = rewards * plugin 

            if cuda:
                rewards = rewards.cuda()
            prob,_,_ = generator.forward(inputs)            
            loss = gen_gan_loss(prob, targets, rewards)
            log=open(log_file+str(flag),"a")
            log.write("Loss value as reward: ")
            log.write(str(loss.data))
            log.write("\n")
            log.close()
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 2 == 0 or total_batch == ADV_STEPS - 1:
            if conditional_flag == 1:
                generate_samples_conditional(generator, encoder, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            else:
                generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            code2text(EVAL_FILE,wi_dict, iw_dict,str(AUG_LOC)+"/"+test_file_loc+str(flag))
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            bleu = eval_epoch_bleu(EVAL_FILE, TRAIN_DATA_LOC,flag,write_file,wi_dict,iw_dict)
            log=open(log_file+str(flag),"a")

            log.write('Batch [%d] BLEU Score: %f' % (total_batch, bleu))
            log.write("\n")
            log.close()
        rollout.update_params()
        return generator,encoder,discriminator
    else:
        # Train the generator for one step
        for it in range(G_STEPS):
            if conditional_flag==1:
                hidden = sample_gen(generator, encoder, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
                samples = generator.sample_conditional(BATCH_SIZE, g_sequence_len, hidden)
            else:
                samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            if mixed_flag ==0:
                rewards = rollout.get_reward(samples, 16, discriminator)
            else:
                samples_tag = text_code2tag_code(samples,wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,VOCAB_SIZE,VOCAB_SIZE_TAG)
                # calculate the reward
                rewards_tag = rollout_tag.get_reward(samples_tag, 16, discriminator_tag)
                rewards_text = rollout.get_reward(samples, 16, discriminator)
                rewards = rewards_tag + rewards_text
            ####
            
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if conditional_flag == 1:
                generate_samples_conditional(generator, encoder, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            else:
                generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            code2text(EVAL_FILE,wi_dict, iw_dict,str(AUG_LOC)+"/"+test_file_loc+str(flag))
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)

            if plugin_flag == "bleu":
                bleu = eval_epoch_bleu(EVAL_FILE, TRAIN_DATA_LOC,flag,write_file,wi_dict,iw_dict)
                bleu = scale(rewards,bleu)
                if plugin_operation == "add":
                    if rew_flag == 0:
                        rewards = rewards - bleu
                    else:
                        rewards = rewards + bleu
                else:
                    if rew_flag == 0:
                        rewards = rewards * (max_val - bleu)
                    else:
                        rewards = rewards * bleu                   
            else:
                plugin = plugin_toolkit(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE, temp_file, label_length,plugin_flag)
                plugin = scale(rewards,plugin)
                if plugin_operation == "add":
                    if rew_flag == 0:
                        rewards = rewards - plugin
                    else:
                        rewards = rewards + plugin
                else:
                    if rew_flag == 0:
                        rewards = rewards * (max_val - plugin)
                    else:
                        rewards = rewards * plugin 

            if cuda:
                rewards = rewards.cuda()
            prob,_,_ = generator.forward(inputs)            
            loss = gen_gan_loss(prob, targets, rewards)
            log=open(log_file+str(flag),"a")
            log.write("Loss value as reward: ")
            log.write(str(loss.data))
            log.write("\n")
            log.close()
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        for count in range(D_STEPS):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE,g_sequence_len)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for count1 in range(EPOCH_IN_DIS):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
                log=open(log_file+str(flag),"a")
                log.write('Loss : %f' % (loss))
                log.write("\n")
                log.close()
                if count%2==0:
                    torch.save(discriminator,  "./"+ dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_dis"+str(count)+str(count1))
                    torch.save(generator,  "./"+ dir_name_for_logs+"/"+DIRECTORY_LABEL+"/model_gen"+str(count)+str(count1))


        if total_batch % 2 == 0 or total_batch == ADV_STEPS - 1:
            if conditional_flag == 1:
                generate_samples_conditional(generator, encoder, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            else:
                generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE,g_sequence_len)
            code2text(EVAL_FILE,wi_dict, iw_dict,str(AUG_LOC)+"/"+test_file_loc+str(flag))
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            bleu = eval_epoch_bleu(EVAL_FILE, TRAIN_DATA_LOC,flag,write_file,wi_dict,iw_dict)
            log=open(log_file+str(flag),"a")

            log.write('Batch [%d] BLEU Score: %f' % (total_batch, bleu))
            log.write("\n")
            log.close()
        rollout.update_params()
        return generator,encoder,discriminator




# Train the data's generator
generator,plot_gan_text = generator_training(PRE_EPOCH_NUM_GEN,SEED_ON_LABEL_EPOCH,generator,gen_data_iter_label,gen_criterion,gen_optimizer,
                       log_file,flag,gen_data_iter,dir_name_for_logs,DIRECTORY_LABEL,BATCH_SIZE,GENERATED_NUM,EVAL_FILE,TRAIN_DATA_LOC,write_file,
                              wi_dict,iw_dict,g_sequence_len)
encoder = copy.deepcopy(generator)


# In[14]:


# Train the TAG's generator 
generator_tag,plot_gan_tag = generator_training(PRE_EPOCH_NUM_GEN_TAG,SEED_ON_LABEL_EPOCH,generator_tag,gen_data_iter_label_tag,gen_criterion_tag,
                                                gen_optimizer_tag,log_file,flag,gen_data_iter_tag,dir_name_for_logs,DIRECTORY_LABEL,BATCH_SIZE,GENERATED_NUM,
                                                EVAL_FILE_TAG,TAG_DATA_LOC,write_file_tag,wi_dict_tag,iw_dict_tag,g_sequence_len_tag)
encoder_tag = copy.deepcopy(generator_tag)



# Initialising the parameters for Discriminator
dis_criterion = nn.NLLLoss(reduction='sum')
dis_optimizer = optim.Adagrad(discriminator.parameters())
dis_criterion_tag = nn.NLLLoss(reduction='sum')
dis_optimizer_tag = optim.Adagrad(discriminator_tag.parameters())
if cuda:
    dis_criterion = dis_criterion.cuda()
    dis_criterion_tag = dis_criterion_tag.cuda()
log=open(log_file+str(flag),"a")
log.write("Pretrain with Discriminator ...")
log.write("\n")
log.close()




# Train the Data's discriminator
print('Pretrain Data Discriminator ...')
discriminator,plot_dis_data = discriminator_training(PRE_EPOCH_NUM_DIS,conditional_flag,generator,encoder,BATCH_SIZE,GENERATED_NUM, 
                                           NEGATIVE_FILE,POSITIVE_FILE, EPOCH_IN_DIS,discriminator, dis_criterion, dis_optimizer,
                                           dir_name_for_logs,DIRECTORY_LABEL,g_sequence_len)


# In[19]:


# Training the TAG's discriminator
print('Pretrain Tag Discriminator ...')
discriminator_tag,plot_dis_tag = discriminator_training(PRE_EPOCH_NUM_DIS_TAG,conditional_flag,generator_tag,encoder_tag,BATCH_SIZE,GENERATED_NUM, 
                                           NEGATIVE_FILE_TAG,POSITIVE_FILE_TAG, EPOCH_IN_DIS,discriminator_tag, dis_criterion_tag, dis_optimizer_tag,
                                           dir_name_for_logs,DIRECTORY_LABEL,g_sequence_len_tag)


# In[ ]:


# Adversarial Training
rollout = Rollout(generator, 0.8)
rollout_tag = Rollout(generator_tag, 0.8)
log=open(log_file+str(flag),"a")
log.write("Start Adversarial Training...")
log.write("\n")
log.close()
print('Start Adversarial Training...\n')
gen_gan_loss = GANLoss()
gen_gan_optm = optim.Adam(generator.parameters())
gen_gan_optm_tag = optim.Adam(generator_tag.parameters())
if cuda:
    gen_gan_loss = gen_gan_loss.cuda()
gen_criterion = nn.NLLLoss(reduction='sum')
gen_criterion_tag = nn.NLLLoss(reduction='sum')
if cuda:
    gen_criterion = gen_criterion.cuda()
    gen_criterion_tag = gen_criterion_tag.cuda()
dis_criterion = nn.NLLLoss(reduction='sum')
dis_criterion_tag = nn.NLLLoss(reduction='sum')
dis_optimizer = optim.Adam(discriminator.parameters())
dis_optimizer_tag = optim.Adam(discriminator_tag.parameters())
if cuda:
    dis_criterion = dis_criterion.cuda()
    dis_criterion_tag = dis_criterion_tag.cuda()


# In[ ]:


# Adversarial Training for Generator and Discriminator in data i.e, G--->D or D---->G
for total_batch in range(ADV_STEPS):
    generator,encoder,discriminator = adversarial_training(generator,encoder,discriminator,generator_tag,encoder_tag,discriminator_tag,TRAIN_DATA_LOC,
                                                                 BATCH_SIZE,GENERATED_NUM,NEGATIVE_FILE,g_sequence_len,G_STEPS,D_STEPS,ADV_STEPS,EVAL_FILE,
                                                                 wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,AUG_LOC,test_file_loc,flag,write_file,max_val,log_file,dis_criterion,
                                                                 dis_optimizer,gen_gan_optm,conditional_flag,total_batch,gen_dis_flag,rollout,POSITIVE_FILE,VOCAB_SIZE,
                                                                 VOCAB_SIZE_TAG,0)



# Adversarial Training for Generator and Discriminator in TAG i.e, G--->D or D---->G
for total_batch in range(ADV_STEPS_TAG):
    generator_tag,encoder_tag,discriminator_tag = adversarial_training(generator_tag,encoder_tag,discriminator_tag,generator,encoder,discriminator,TAG_DATA_LOC,BATCH_SIZE,
                                                                       GENERATED_NUM,NEGATIVE_FILE_TAG,g_sequence_len_tag,G_STEPS,D_STEPS,ADV_STEPS_TAG,
                                                                       EVAL_FILE_TAG,wi_dict_tag,iw_dict_tag,wi_dict,iw_dict,AUG_LOC,test_file_loc_tag,flag,write_file_tag,max_val,log_file,
                                                                       dis_criterion_tag,dis_optimizer_tag,gen_gan_optm_tag,conditional_flag,total_batch,gen_dis_flag,rollout_tag,
                                                                       POSITIVE_FILE_TAG,VOCAB_SIZE_TAG,VOCAB_SIZE,0)


# In[ ]:


# Adversarial Training for Generator and Discriminator in TAG i.e, G--->D or D---->G
for total_batch in range(ADV_STEPS_FINE):
    generator,encoder,discriminator = adversarial_training(generator,encoder,discriminator,generator_tag,encoder_tag,discriminator_tag,TRAIN_DATA_LOC,
                                                                 BATCH_SIZE,GENERATED_NUM,NEGATIVE_FILE,g_sequence_len,G_STEPS,D_STEPS,ADV_STEPS,EVAL_FILE,
                                                                 wi_dict,iw_dict,wi_dict_tag,iw_dict_tag,AUG_LOC,test_file_loc,flag,write_file,max_val,log_file,dis_criterion,
                                                                 dis_optimizer,gen_gan_optm,conditional_flag,total_batch,gen_dis_flag,rollout,POSITIVE_FILE,VOCAB_SIZE,
                                                                 VOCAB_SIZE_TAG,1)



