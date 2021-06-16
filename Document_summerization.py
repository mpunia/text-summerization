from flask import Flask, render_template, request
import math
import re
import nltk
from nltk.corpus import stopwords
import string 
stopwords = set(stopwords.words('english'))
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
stopwords = set(stopwords.words("english"))
import pickle

#text = r"E:\projects\text summerization end to end\wikiAI.txt"
#text = open(text , encoding="utf8")
#text = text.read()

text = str(input('paste your text here  '))
sentence = sent_tokenize(text)


def clean_text(text):
    sent_freq = {}
    sentence = re.sub(r"http\S+", "", text)
    sentence = re.sub(r'[^a-zA-Z0-9 _,!.s]' , '' , sentence)
    sentences = sent_tokenize(sentence)

#   return sentences
#sentences = clean_text(text)
#len_sentences = len(sentences)
 

    for sent in sentences:
        words = sent.split(' ')
        words = [word for word in words if word not in string.punctuation]
        freq_table = {}
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopwords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        sent_freq[sent] = freq_table
    return sent_freq

#freq_matrix = clean_text(text)

#total_documents=  len(freq_matrix)

def tf_matrix(freq_matrix):
    
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        
        tf_table = {}
        count_words_in_sentence =  len(f_table)

        for word, count in f_table.items():
            
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

tf = tf_matrix(freq_matrix)

filename3 = 'tf_matrix.pkl'
pickle.dump(tf , open(filename3 , 'wb'))

def documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

doc_per_words_count = documents_per_words(freq_matrix)

filename4 = 'doc_per_words_count.pkl'
pickle.dump(doc_per_words_count  , open(filename4 , 'wb'))

def idf_matrix(freq_matrix, doc_per_words_count, len_sentences):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(len_sentences / float(doc_per_words_count[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

idf = idf_matrix(freq_matrix, doc_per_words_count, total_documents)

filename5 = 'idf.pkl'
pickle.dump(idf , open(filename5 , 'wb'))

def tf_idf_matrix(tf , idf):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf.items() , idf.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),f_table2.items()): 
            # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

tf_idf_matrix = tf_idf_matrix(tf , idf)

filename6 = 'tf_idf_matrix.pkl'
pickle.dump(tf_idf_matrix , open(filename6 , 'wb'))


def score_sentences(tf_idf_matrix):
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

sentence_scores = score_sentences(tf_idf_matrix)

filename7 = 'sentence_scores.pkl'
pickle.dump(sentence_scores , open(filename7 , 'wb'))

def average_score(sentence_scores):
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for score in sentence_scores:
        sumValues += sentence_scores[score]

    # Average value of a sentence from original text
    average = (sumValues / len(sentence_scores))

    return average
threshold = average_score(sentence_scores)

filename8 = 'threshold.pkl'
pickle.dump(threshold , open(filename8 , 'wb'))

def generate_summary(sentence_scores , threshold):
    sentence_count = 0
    summary = ''
    for sentence,score in sentence_scores.items():
        if score >= threshold:
            summary += sentence
            
    return summary


summary = generate_summary(sentence_scores , threshold)


print(summary)

filename = 'document summerization.pkl'
pickle.dump(summary , open(filename, 'wb'))



