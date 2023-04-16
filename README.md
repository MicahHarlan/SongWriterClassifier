# SongWriterClassifier

!!!IMPORTANT!!! 
Some of the programs will not work, unless you download: https://nlp.stanford.edu/projects/glove/ specifically the file named "glove.6B.50d.txt". This text file is too big for github

This program uses machine learning to classify song lyrics into three categories: Nirvana, The Beatles, and Black Sabbath. It uses the spacy, keras, and numpy libraries.

Usage

To use this program, you must have three text files, each containing the lyrics for one of the three artists: "Nirvana.txt", "beatles.txt", and "BlackSabbath.txt". Place these files in the same directory as the program.

Run the program and enter a sentence to classify it as belonging to one of the three artists. Repeat until you are finished.

Dependencies

This program requires the following libraries:

spacy
keras
numpy


DEFINITIONS:
bow: train a neural net using the bag of words model of NLP.
Seq: uses neural net trained on word embeddings
