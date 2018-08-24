__author__ = 'Rose'
import numpy as np
import pandas as pd
import os, sys
from convert_pdf import convert
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from math import log
import operator
import re


document_vector_dict = {}
frequency_array = []

class DocumentVector:
    
    """
    Create a frequency vector for a single document 
    tf = term frequency, used for calculating tf-idf weights
    """

    def __init__(self, filename=None):
        self.filename = filename # for column headers in matrix
        self.words = self.build_corpus() # tokens in document, excluding stopwords
        self.frequencies = self.get_word_frequencies() # frequency dict
        tf_tup = self.calculate_tf() # term frequency/vocab size
        self.tf_values = tf_tup[0] # tf value
        self.tf_labels = tf_tup[1] # dict of tuples: (term, tf)
        #frequency_array.append(pd.Series(self.frequencies, name=filename))
        #document_vector_dict[filename] = self.tf_labels # dict of tuples: (filename, labelled tf dict)

    def __repr__(self):
        print(self.frequencies)

    def build_corpus(self):
        text_file = self.filename
        reload(sys)
        sys.setdefaultencoding('utf8')
        file_location = os.path.join('/home/rose/solirius-profiles', text_file)
        #try:
        text = convert(file_location)
        #except:
          #  print('failed to find file', file_location)
        # text.decode('utf-8')
        text = text.strip()
        tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')  # get rid of backslashes
        stop_words = set(stopwords.words('english'))
        tokens = tokenizer.tokenize(text.lower())
        words = [t for t in tokens if t not in stop_words]
        del stop_words
        return words

    def vocab_size(self):
        return len(self.words)
    
    def token_count(self):
        return len(self.words)

    def type_count(self):
        return len(self.frequencies)

    def print_n_highest_freqs(self, n):
        sorted_freqs = sorted(self.frequencies.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_freqs[0:n])

    def get_word_frequencies(self):
        word_freqs = defaultdict(int)
        for word in self.words:
            if word not in word_freqs:
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
        return word_freqs
    
    def calculate_tf(self):
        v_count = self.vocab_size()
        tf_matrix = np.zeros(self.token_count())
        tf_labels = defaultdict(int)
        count = 0
        for term_tup in self.frequencies.items():
            tf = float(term_tup[1])/float(v_count)
            tf_matrix[count] = tf
            count += 1
            tf_labels[term_tup[0]] = tf
        return tf_matrix, tf_labels

    def get_freqs_from_list(self, wordlist):
        """
        Method for creating a series object of word frequencies for a document
        but using an already existing list of words, with 0 for words that do not occur.
        This is so that the test data frequency vector matches the training data frequency matrix,
        otherwise we don't have a word:word correspondence and we also have dimensionality issues.
        """
        wordlist_dict = defaultdict(int)
        for word in wordlist:
            if word in self.frequencies:
                wordlist_dict[word] = self.frequencies[word]
            else:
                wordlist_dict[word] = 0
        freqs_array = pd.Series(wordlist_dict)
        return freqs_array


class TermDocumentMatrix:

    """
    Class for creating a matrix which contains tf-idf weights for a set of documents
    tf-idf weights (term frequency - inverse document frequency) are a measure of how important
    a word is to a document/class of documents, based on word frequency and normalised by document length
    and rarity across multiple documents
    """

    def __init__(self, vector_dict):
        self.tf_matrix = pd.DataFrame(vector_dict)
        self.num_docs = len(vector_dict)
        self.tf_idf = self.build_tf_idf_matrix()
        
    def __repr__(self):
        print(self.tf_matrix)

    def calculate_idf(self):
        term_doc_count = self.tf_matrix.count(axis=1, numeric_only=False)
        term_idf = self.num_docs/term_doc_count
        idf_values = term_idf.map(lambda x: log(x))
        return idf_values

    def build_tf_idf_matrix(self):
        tf = self.tf_matrix
        idf = self.calculate_idf()
        tf_idf_matrix = tf.div(idf.iloc[0], axis=1)
        return tf_idf_matrix

    def as_array(self):
        return self.tf_idf.values


class FrequencyMatrix:

    def __init__(self, freqs_dict):
        frequency_matrix = pd.DataFrame(freqs_dict).transpose()
        frequency_matrix.fillna(0, inplace=True)
        self.frequency_matrix = frequency_matrix
        self.num_docs = len(freqs_dict)

    def as_array(self):
        return self.frequency_matrix.values



# vector_one = DocumentVector('Alexander Juggins Profile.pdf')
# vector_two = DocumentVector('Alec Doran-Twyford Profile.pdf')
# vector_three = DocumentVector('Alistair Tooke Solirius profile.pdf')

#matrix = TermDocumentMatrix(document_vector_dict)
#print(matrix.tf_idf.head())
#print(matrix.tf_matrix)

#matrix_f = FrequencyMatrix(frequency_array)
#print(matrix_f.frequency_matrix)
#print(matrix_f.head())

