# Sentiment-Classification 

## Features Engineering

## Word2Vec : CBOW and skipgram
As some of the words or terms relating to Covid-19 are relatively new, word embeddings were being trained from scratch using Continuous Bag of Words (“CBOW”) model and Skipgram Model  to capture contextual and semantic similarity. The cleaned dataset which contained more than 5000 tweets were used to build the corpus vocabulary. The tweets were tokenised and each unique word token was mapped to a unique identification number and a “PAD” term was introduced to pad context words to a fixed length if required. The vocabulary size amounted to 12,649.

### GLoVe

Trained word embeddings using either the CBOW model or skipgram model are limited to the vocabulary corpus which was used for training. As such, additional pre-trained word vectors from Global Vectors for Word Representation (“GloVe”) were also leveraged on. As the dataset is primarily extracted from Twitter, glove.twitter.27B.zip file  was downloaded. It contains 2 billion tweets, 27 billion tokens and 1.2 million vocabularies. The pretrained word vectors of 100d was extracted.

### TF-IDF
For Maximum Entropy and Support Vector Machine (“SVM”) supervised machine learning method used in training the sentiment classifier, the tweets were vectorised using TF-IDF, with variations of N-grams (i.e. unigram and bigram). As the feature document matrix lies in high-dimensional spaces with data sparsity issue and risk of overfitting due to inclusion of irrelevant “noise” features, the top 600 best features were selected from N-grams.

## CPU method
Maximum entropy and SVM models  were the two CPU-based machine learning method used. The features used were obtained by concatenating the top 600 best features from N-gram, self-trained CBOW word embeddings, self-trained Skipgram word embeddings and pre-trained GLoVE twitter word vectors.

## DNN method
Bidirectional-LSTM (Bi-LSTM) models  were also built using the CBOW, Skipgram and GloVe word-embeddings for text classification. Each word-embedding was fitted into a separate Bi-LSTM model. A total of 3 Bi-LSTM models were built, namely, Bi-LSTM (CBOW), Bi-LSTM (Skipgram) and Bi-LSTM (GloVe).


