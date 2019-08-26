#!/usr/bin/python3
import os
import sys
import json
from pathlib import Path
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pycrfsuite

# Define Feature Functions
def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for every words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # features for words which is not the beginning of a decument
    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    # feature for a word which is the beginning of a document
    else:
        features.append('BOS')


    # features for words which is not the end of a decument
    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    # feature for a word which is the end of a document
    else:
        features.append('EOS')

    return features

# Load dataset
def load_dataset(input_file):

    # load input_file
    if not os.path.exists(input_file):
        print("There is no such file named '%s'" %(input_file))
        exit(1)

    try:
        raw_data = json.loads(Path(input_file).read_text(), encoding='utf-8')
        for data in raw_data:
            temp_list = [(t[0], t[1]) for t in data]
            docs.append(temp_list)
    except:
        print("Input file is not a List of String")
        exit(1)

    if os.path.exists(output_file):
        print("'%s' already exists. Are you sure you will use that name?" %(output_file))
        response = input('[Y / N] : ')
        if response.lower() != 'y':
            exit(1)


# Generate POS Tags
def generate_POS_Tags():
    #data = []
    for i, doc in enumerate(docs):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]

        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)

        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

# Make training set
def make_training_set():
    # generate features for a document
    def extract_features(doc):
        return [word2features(doc, i) for i in range(len(doc))]

    # get labels of each token in a document
    def get_labels(doc):
        return [label for (token, postag, label) in doc]

    X = [extract_features(doc) for doc in data]
    y = [get_labels(doc) for doc in data]

    return X, y

def train_model():
    trainer = pycrfsuite.Trainer(verbose=False)  # verbose = False (-> do not print the process of training)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })

    # set the file name of model, which will be saved
    trainer.train(output_file)

def show_result():
    tagger = pycrfsuite.Tagger()
    tagger.open(output_file)
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    # ========= find all the classes in testSet ===========
    n = 0
    all_labels_dict = {}
    all_labels_list = []
    for labels in y_pred :
        for label in labels:
            if label not in all_labels_list:
                all_labels_list.append(label)
                all_labels_dict[label] = n
                n += 1
    print("LABELS : ", all_labels_dict)

    # =====================================================

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([all_labels_dict[tag] for row in y_pred for tag in row])
    truths = np.array([all_labels_dict[tag] for row in y_test for tag in row])

    # Print out the classification report
    print(classification_report(
        truths, predictions,
        target_names = all_labels_list)
    )

if __name__ == '__main__':
    docs = []
    data = []

    # ======== input_file & output_file exception ==========
    if len(sys.argv) != 3:
        print("Please enter the name of 'input_file' and 'output_model'")
        exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] + '.model'

    load_dataset(input_file)
    # ======================================================

    generate_POS_Tags()
    X, y = make_training_set()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_model()
    show_result()
