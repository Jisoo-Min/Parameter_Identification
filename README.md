# Parameter-Identification

For a long time, many people want to generate code from sentences automatically. Thanks to machine learning and deep learning, we can try to deal with this problem. 
As part of efforts to automatically generate code, we predict which parameters will be use in the function when a sentence is given. 

To identify parameters, we use the slot filling approach, which aims to extract the values of certain types of attributes for a given entity from a large collection of source documents.

For instance, in airline information, we know which words mean departure and which words mean time. 
```text
Show flights from Boston to New York today.
```
In this example, Boston is a point of departure, New York is an arrival, and Today is the time. To slot filling, we define slots using IOB presentation. 


| Show | flights | from | Boston      | to | New       | York      | today. |
|------|---------|------|-------------|----|-----------|-----------|--------|
| O    | O       | O    | B-departure | O  | B-arrival | I-arrival | B-time |


In this scheme, each word is tagged with one of three tags, I (inside), O (outside), or B (begin). 
if a token is the beginning of a chunk, it is tagged B such as Boston, New, and today. Also, Subsequent tokens within the chunk are tagged I like york. Other undefined words have O tag

We identify parameters using RNN and CRF to do slot filling.


## RNN <br/>
RNN runs through four phases: data processing, model generation, training, and evaluation
1. Data Processing<br/>
We have Codingbat dataset we made. It has queries and correspponing slot names. Because the data we made were divided into two files (word with slot ID and slot name with slot ID), the process is binding them together.

| Given | an      | array   | of      | int,    | return | the | sum | of | all | the | elements. |
|-------|---------|---------|---------|---------|--------|-----|-----|----|-----|-----|-----------|
| O     | B-param | I-param | I-param | I-param | O      | O   | O   | O  | O   | O   | O         |

At the end of the process, we can get two sets of vectors: one is a set of query vectors embedded with one-hot encoding and the others is a set of vectors made of slot IDs.

Query vector = [734, 379, 1089, 814, 957, 1149, 1374, 776,  831, 573, 379, 375, 615, 1000, 1395, 188, 1358, 733, 1334, 1156, 614, 414, 261, 1358, 622, 1334, 1156, 615],<br/>
Slot vector  = [11, 11, 3, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]

2. Model generation<br/>
With several settings, instanciate the model. For example, it takes the number of hidden layers, context-size, dimension of word embedding, etc.
We use two RNN model: Elman-RNN and Jordan-RNN

3. Training <br/>
The training operate according to the epochs and the number of training data. 
For each epoch, the best f1-score is recorded. 

4. Evaluation <br/>
During training, the best f1-score is recorded. At the end of the training, the best f1-score is calculated in testset.

#### How to run
To run this code, python3 must be installed. In addition, python libraries (numpy, theano, and keras) are required. See [requirements](https://github.com/jiisoo/Parameter-Identification/blob/master/RNN/requirements.txt) in RNN folder.

In parent folder of root directory, <br/>
Run Elman-RNN.<br/>
```shell
python3 Parameter-Identification/RNN/examples/elman-forward.py
```

Run Jordan-RNN.<br/>
```shell
python3 Parameter-Identification/RNN/examples/jordan-forward.py
```



## CRF
