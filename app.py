import pickle
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from flask import Flask, render_template, request
import math
import re
import nltk
from nltk.corpus import stopwords
import string
stopwords = nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()


#text = str(input('paste your text here  '))
#sentence = sent_tokenize(text)


def clean_text(text):
    freq_matrix = {}
    #sentences = sent_tokenize(text)
    sentence = re.sub(r'^https?:\/\/*[\r\n]*', '', str(text))
    sentence = re.sub(r'[^a-zA-Z0-9 _,!.s]', '', sentence)
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
        freq_matrix[sent] = freq_table

    len_sentences = len(freq_matrix)

    tf_matrix = {}

    for sent, f_table in freq_matrix.items():

        tf_table = {}
        count_words_in_sentence = len(f_table)

        for word, count in f_table.items():

            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table


#tf = tf_matrix(freq_matrix)

    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1


#doc_per_words_count = documents_per_words(freq_matrix)

    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(
                len_sentences / float(word_per_doc_table[word]))

        idf_matrix[sent] = idf_table


#idf = idf_matrix(freq_matrix, doc_per_words_count, total_documents)

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table


#tf_idf_matrix = tf_idf_matrix(tf , idf)

    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentence_scores = {}

    for sent, f_table in tf_idf_matrix.items():

        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():

            total_score_per_sentence += score

        sentence_scores[sent] = total_score_per_sentence / \
            count_words_in_sentence


#sentence_scores = score_sentences(tf_idf_matrix)

    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for score in sentence_scores:
        sumValues += sentence_scores[score]

    # Average value of a sentence from original text
    threshold = (sumValues / len(sentence_scores))


#threshold = average_score(sentence_scores)

    sentence_count = 0
    summary = ''
    for sentence, score in sentence_scores.items():
        if score >= threshold:
            summary += sentence

    return summary


#summary = clean_text(text)

app = Flask(__name__)


@app.route('/', methods=['GET' , 'POST'])
def home():
    return render_template('home.html')
@app.route('/Back_btn')
def back():
    return render_template(("home.html"))

@app.route('/summary',methods=['GET', 'POST'])
def summary():
    if request.method == 'POST':
        message = request.form.get('message')

        summary = clean_text(message)

        return render_template('summary.html',summary=summary)
    return render_template("homes.html")


if __name__ == "__main__":
    app.run(debug=True)
