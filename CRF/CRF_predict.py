import sys
import os
import json
from pathlib import Path
import nltk
import pycrfsuite

# Load dataset
def load_dataset():
    # Load model_file
    try:
        tagger.open(model_name)
    except:
        print("There is no such file named '%s'" %(model_name))
        exit(1)

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

# generate features for a document
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# Future work : expand input from (string) to (list of strings)
def preprocess_sentence(sentence):
    tokens = sentence.split()
    pos_tagged = nltk.pos_tag(tokens)

    X = extract_features(pos_tagged)
    return X


def predict(X):
    return tagger.tag(X)

def extract_parameters(X, y):

    sentence = ''

    p_temp_arg = '';
    p_prev_label = 'O'

    p_args = [];

    for i in range(len(y)):
        word = X[i][1].split("=")[1]
        p_label = y[i]
        #print("Word : %s, Previous Label : %s, Present Label : %s" %(word, p_label, p_prev_label))

        # =========predicted arguments =================
        if p_label == 'O' or p_label == 'X':
            if p_prev_label != 'O' and p_prev_label != 'X':
                p_args.append(p_temp_arg)
                p_temp_arg = ''
            else:
                pass
        else:
            if p_prev_label == 'O' or p_prev_label == 'X':
                p_temp_arg += (word + ' ')
            else:
                if (p_prev_label[0] == 'B' or p_prev_label[0] == 'I') and p_label[0] == 'I':
                    p_temp_arg += (word + ' ')
                else:
                    p_args.append(p_temp_arg)
                    p_temp_arg = (word + ' ')
        p_prev_label = p_label
        # ===========================================
        sentence += (word + ' ')
    print(" [ Sentence ] : ", sentence)
    print(" [ Predicted Arguments ] : ", p_args)


if __name__ == '__main__':
    # ======== input_file & output_file exception ==========
    if len(sys.argv) != 2:
        print("Please enter the 'CRF model'")
        exit(1)

    model_name = sys.argv[1] # .model

    # File name correction
    if len(model_name) <= 6 or (len(model_name) > 6 and model_name[-6:] != '.model'):
        model_name = model_name + '.model'

    input_sentence = input("Enter the sentence : ")

    tagger = pycrfsuite.Tagger()
    load_dataset()

    X = preprocess_sentence(input_sentence)
    y = predict(X)

    extract_parameters(X, y)
    # ======================================================