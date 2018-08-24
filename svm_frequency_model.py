__author__='Rose'
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import nltk

from nlp import DocumentVector, TermDocumentMatrix, FrequencyMatrix

"""
This file contains an implementation of an SVM model 
using word frequency vectors as input.
"""

#'nltk.download('stopwords')

frequency_array = []
document_vector_dict = {}

# Training data

positive_examples = [] # put positive examples here

negative_examples = [] # put negative examples here

# Prepare labels matrix
Y_matrix = np.zeros(shape=(len(positive_examples+negative_examples), 1))
count = 0

# Create training data
for p in positive_examples:
    vector = DocumentVector(p)
    frequency_array.append(pd.Series(vector.frequencies, name=p))
    document_vector_dict[p] = vector.tf_labels
    Y_matrix[count, 0] = 1
    count += 1

for n in negative_examples:
    vector = DocumentVector(n)
    frequency_array.append(pd.Series(vector.frequencies, name=n))
    document_vector_dict[n] = vector.tf_labels
    Y_matrix[count, 0] = 0
    count += 1

# Initialize frequency matrix
f_matrix = FrequencyMatrix(frequency_array)
#print(f_matrix.frequency_matrix.head())

# Prepare SVC model input matrices
X = f_matrix.frequency_matrix.transpose().values
Y = Y_matrix.ravel()

# for getting test data vector
word_list = f_matrix.frequency_matrix.index.values

# Test data
test_file = '' # test instances go here

# Prepare test data
test_vector = DocumentVector(test_file)
test_frequencies = test_vector.get_freqs_from_list(word_list)
test_frequencies = test_frequencies.values.reshape(-1, 1).transpose()
#print(test_frequencies.shape, type(test_frequencies))

#matrix = TermDocumentMatrix(document_vector_dict)
#print(matrix.tf_idf.head())

# Train model
clf = SVC(probability=True)
clf.fit(X, Y)
#params = pickle.dumps(clf)

# Predict
prediction= clf.predict(test_frequencies)
suit_score = clf.predict_proba(test_frequencies)
print(prediction[0])
print(suit_score)
print "Candidate Suitability Score:", suit_score[0][0]*100
