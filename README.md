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


## RNN


## CRF
