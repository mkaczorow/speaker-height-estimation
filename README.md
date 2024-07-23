# Speaker height estimation

This repository is used to implement a program that estimates speaker height using neural networks.

Work is based on the article: "End-to-End Speaker Height and age estimation using Attention Mechanism with LSTM-RNN" written by: Manav Kaushik, Van Tung Pham, Eng Siong Chng.
https://arxiv.org/pdf/2101.05056

### database
TIMIT dataset was used in this project to create model for estimate speaker's height. Corpus contains 5 hours of English speech, was designed to provide speech data for acoustic-phonetic studies and evaluation of automatic speech recognition systems. Recording include clips of 630 (70% men, 30% female) speakers of 8 dialects of American English, where each reading 10 sentences.

The default test set contains recordings of 2 male and 1 female speakers from each dialect, what gives corpus of 24 speakers. 


### Repository files:
* features.py 
This file contains functions to extract features from audio, acoustic features of 80 filetr bank energies and 3 pitch features.
* 
