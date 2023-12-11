from pathlib import Path
import docx2txt
import numpy as np
import pandas as pd
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('vader_lexicon')



# Preprocessing Coding Functions
#These are helper functions to:  
#(1) to open an individual word file (transcript)   
#(2) tokenize text in list of sentences, remove sentences spoken by FMSS administrator   
#(3) tokenize sentences in lists of words, remove behavioral descriptions of participant and punctuation  
#(4) rejoin the list of words back into sentences  
#(5) append an ID column to list of sentences

def file_opener(data_folder, filename):
    '''Helper function. Input is file location. Output is text of that file.'''
    filename = data_folder / filename   
    text = docx2txt.process(filename)   
    
    return text


def preprocess(text):
    '''Helper function. Converts text into lower case, tokenizes into sentences, 
    and removes "administrator:" sentences.
    Input is text, ouput is tokenized, lower-case sentences, without admin sentences'''    
    text = text.lower()    
    sentences_raw = nltk.sent_tokenize(text)    
    sentences = []    

    for i in range(len(sentences_raw)):
        if 'administrator:' in sentences_raw[i-1]:
            pass
        else:
            sentences.append(sentences_raw[i-1])   

    return sentences


def remove_bracketed_words(words):
    open_bracket = "("
    close_bracket = ")"
    start_index = words.index(open_bracket)
    stop_index = words.index(close_bracket) + 1
    del words[start_index : stop_index]

    return words


def sentence_cleaner(sentences):
    '''Helper function. Goes through each sentence, removes bracketed descriptions, and removes stopwords.
    Input is list of sentences. Output is list of cleaned sentences'''   
    sentences_cleaned = []    
    stopwords = ['participant', '_____', ':', 'umm…', 'ummm…', 'uhm…']  

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        
        new_words = [word for word in words if word not in stopwords]
        
        while new_words.count(")") > 0 and new_words.count("(") > 0:
            words_cleaned = remove_bracketed_words(new_words)
        else:
            words_cleaned = new_words
        
        words_cleaned_nopunct = [word for word in words_cleaned if word.isalpha()]
            
        sentences_cleaned.append(words_cleaned_nopunct) 

    return sentences_cleaned



def sentence_joiner(sentences_cleaned):  
    sentences_joined = []   

    for sentence in sentences_cleaned:
        words_joined = ' '.join(sentence)
        sentences_joined.append(words_joined)  

    return sentences_joined


def id_column(filename, sentences):
    '''Helper function. Creates an ID list for the file. Input is the filename and tokenized list of sentences.
    Output is column of filename repeated for length of sentences''' 
    ID = []
    filename_cleaned = filename.split(".docx")[0] 

    for i in range(len(sentences)):
        ID_instance = filename_cleaned
        ID.append(ID_instance)  

    return ID

# VADER Coding Function, Word and Sentence Count
#There are helper functions to:  
#(1) apply sentiment scores to each sentence in the text  
#(2) count number of words in each sentence in the text  

def sentiment_score(sentences_joined):
    '''Helper function. Scores the sentiment of each sentence using VADER. Input is tokenized sentences.
    Output is senitment scores (negative, neutral, positive, compound)'''  
    vds = SentimentIntensityAnalyzer()
    sentiment_texts = []

    for sentence in sentences_joined:
        sentiment_text = vds.polarity_scores(sentence)
        sentiment_texts.append(sentiment_text)   

    return sentiment_texts

def count_words_sentence(sentences_joined):
    '''Helper function. Counts the number of words in the cleaned sentence.
    Input is list of cleaned sentences. Output is list of count of words in each sentence'''   
    num_words_list = []
    for s in sentences_joined:
        num_words = len(s.split())
        num_words_list.append(num_words) 

    return num_words_list# Combining Functions to Create VADER and Count Scores Per Transcript



# Combining Functions to Create VADER and Count Scores Per Transcript
#This is the main function to:  
#(1) take file convert into dataframe with sentiment scores and word count for each sentence in text   
#(2) get descriptive statistics of sentiment scores and word counts for all sentences in the text 

def make_dataset(data_folder, filename):
    '''Main function. Input is file location and name. Output is dataframe with rows 
    for each sentence and columns for filename ID, sentence, VADER sentiment codes 
    (negative, neutral, positive, compound), and word counts'''
    
    # dataframe for sentences
    text = file_opener(data_folder, filename)   
    sentences = preprocess(text)  
    sentences_cleaned = sentence_cleaner(sentences)  
    sentences_joined = sentence_joiner(sentences_cleaned) 
    num_words_list = count_words_sentence(sentences_joined)
    ID = id_column(filename, sentences_joined)
    df_sentences = pd.DataFrame({
        'ID' : ID,
        'sentences' : sentences_joined,
        'num_words' : num_words_list,
    })
    # dataframe for sentiment scores
    sentiment_texts = sentiment_score(sentences_joined)    
    df_sentiment = pd.DataFrame(sentiment_texts)   
    # combining sentences and sentiment scores into one dataframe   
    df = pd.concat([df_sentences, df_sentiment], axis=1, sort=False)  

    return df

def get_descriptives(df):
    '''Provides the min, max, mean, and sd for neg, neu, pos, and compound scores.
    Input is the dataframe with tokenized sentences and sentiment scores. 
    Output is the descriptive statistics for the sentiment scores.'''    
    ID = df.iloc[0,0]    
    mean_compound = df['compound'].mean()
    std_compound = df['compound'].std()   
    mean_positive = df['pos'].mean()
    std_positive = df['pos'].std()  
    mean_negative = df['neg'].mean()
    std_negative = df['neg'].std()   
    num_sentences = (len(df['sentences']) + 1) 
    total_words = df['num_words'].sum()
    mean_words = df['num_words'].mean()
    std_words = df['num_words'].std()

    return ID, mean_compound, std_compound, mean_positive, std_positive, mean_negative, std_negative, num_sentences, total_words, mean_words, std_words


def transform(score):
    labels = [label['label'] for label in score]
    scores = [label['score'] for label in score]
    label_map = {'LABEL_0':'neg','LABEL_1':'pos'}
    labels = [label_map[label] if label in label_map.keys() else label for label in labels]
    return {label:score for label,score in zip(labels,scores)}    

def make_dataset_nn(data_folder, filename, classifier):
    '''replicates make_dataset function but for output of hugging face neural net models'''
    
    # dataframe for sentences
    text = file_opener(data_folder, filename)   
    sentences = preprocess(text)  
    sentences_cleaned = sentence_cleaner(sentences)  
    sentences_joined = sentence_joiner(sentences_cleaned) 
    num_words_list = count_words_sentence(sentences_joined)
    ID = id_column(filename, sentences_joined)
    df_sentences = pd.DataFrame({
        'ID' : ID,
        'sentences' : sentences_joined,
        'num_words' : num_words_list,
    })
    # dataframe for sentiment scores
    sentiment_texts = classifier(sentences_joined)
    sentiment_texts = [transform(score) for score in sentiment_texts]
    df_sentiment = pd.DataFrame(sentiment_texts)   
    # combining sentences and sentiment scores into one dataframe   
    df = pd.concat([df_sentences, df_sentiment], axis=1, sort=False)  

    return df

def get_year_descriptives(year_path):
    stats_all_files = []
    data_folder = Path(year_path)
    filelist = os.listdir(year_path)
    for i in filelist:
        if i.endswith(".docx"):
            df = make_dataset(data_folder, i)
            stats = get_descriptives(df)
            stats_all_files.append(stats)
    return pd.DataFrame.from_records(stats_all_files, columns = ['ID', 'mean_compound', 'std_compound',
                                                           'mean_positive', 'std_positive',
                                                           'mean_negative', 'std_negative',
                                                           'num_sentences', 'total_words', 'mean_words','std_words'])

def get_year_descriptives_nn(year_path,classifier):
    stats_all_files = []
    data_folder = Path(year_path)
    filelist = os.listdir(year_path)
    for i in filelist:
        if i.endswith(".docx"):
            df = make_dataset_nn(data_folder, i, classifier)
            stats,column_names = get_descriptives_nn(df)
            stats_all_files.append(stats)
    return pd.DataFrame.from_records(stats_all_files, columns=column_names)


def get_descriptives_nn(df):
    '''Provides the min, max, mean, and sd for neg, neu, pos, and compound scores.
    Input is the dataframe with tokenized sentences and sentiment scores. 
    Output is the descriptive statistics for the sentiment scores.'''    
    ID = df.iloc[0,0]
    num_sentences = (len(df['sentences']) + 1) 
    total_words = df['num_words'].sum()
    mean_words = df['num_words'].mean()
    std_words = df['num_words'].std()
    stats = []
    sentiments = list(df.columns[3:])
    column_headers = []
    for sentiment in sentiments:
        column_headers += [f'mean_{sentiment}', f'std_{sentiment}']
        stats += [df[sentiment].mean(), df[sentiment].std()]  

    stats = [ID] + stats + [num_sentences, total_words, mean_words, std_words]
    headers = ['ID'] + column_headers + ['num_sentences', 'total_words', 'mean_words', 'std_words']
    return stats, headers


def get_sentences(data_folder,filename):
    #one transcript
    data_folder = Path(data_folder)
    text = file_opener(data_folder, filename)
    sentences = preprocess(text)
    sentences_cleaned = sentence_cleaner(sentences)
    return sentence_joiner(sentences_cleaned)
