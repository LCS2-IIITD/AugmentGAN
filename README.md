# AugmentGAN

.................................................... Data Augmentation for Text using GAN ............................................

Some important details about repo :
1. File named as main_mixed.py is main file to run the data augmentation pipeline.
It needs a json file as input which contains : 
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

Sample json file for reference is given as "sample_input_json.json".

2. main_run.sh is script to run the pos and threshold based methods (These are our baseline.)
3. Other notebook (.pynb) and python files are auxiliary files for classification using RNN, Transformer, CNN, LSTM, etc and preprocessing codes.
4. We have some generated data from our proposed gan for references.

References for code:
1. https://github.com/bentrevett/pytorch-sentiment-analysis
2. https://github.com/samhavens/NLP-data-augmentation
3. https://github.com/kumar-shridhar/Know-Your-Intent
4. https://www.kaggle.com/uciml/sms-spam-collection-dataset

	
