# Speaker height estimation

This repository is used to implement a program that estimates speaker height using neural networks.

Project involves creating a model using the LSTM network with an attention mechanism. The model will be trained and tested on a TIMIT dataset. 

Work is based on the article: "End-to-End Speaker Height and age estimation using Attention Mechanism with LSTM-RNN" written by: Manav Kaushik, Van Tung Pham, Eng Siong Chng.
https://arxiv.org/pdf/2101.05056

### Database
TIMIT dataset was used in this project to create model for estimate speaker's height. Corpus contains 5 hours of English speech, was designed to provide speech data for acoustic-phonetic studies and evaluation of automatic speech recognition systems. Recording include clips of 630 (70% men, 30% female) speakers of 8 dialects of American English, where each reading 10 sentences.

The default test set contains recordings of 168 speakers (112 men, 56 female).

### Model
As in the publication, the model is built on the basis of the lstm network with an attentional layer. at first, from the audio recording, the program extracts features - in the publication it is the filter bank energies and pitch, in my case it is only filter bank energies, because the kaldi program used in the model system for the extraction of pitch features, does not synchronize with the Windows system. Then the features are encoded in front of the long short-term memory network and passed to the attention layer. As a final step, the values are passed to the dense layer to show the speaker's height prediction. 


### Results
|      | LSTM (without attention mechanism) | LSTM (with attention mechanism)|
| ---- | ---------------------------------- | ------------------------------ |
| RMSE | 9.61                               |              9.56              |
| MAE  | 7.76                               |              7.69              |

In the above results as well as in the paper, you can see that the attention mechanism positive affects the results of the experiment. The attention mechanism was introduced to deal with forgetting the parts of long sequences. 

### Repository files:
* features.py
  
This file contains functions to extract features from audio, acoustic features of 80 filetr bank energies and 3 pitch features.

* model_1, max_timestep 

Necessary files for testing the model.

* testing.py

Python file, where you can check how model works of yours audio files.

