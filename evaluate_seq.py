from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from keras.preprocessing.text import Tokenizer

from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Flatten

import numpy as np


nlp = English() #spacy.load("en_core_web_sm") 
t = Tokenizer()
sentencizer = Sentencizer()
np.set_printoptions(precision=2)

with open("BlackSabbath.txt", encoding="utf-8") as f:
	black_sab = f.read()	
	black_sab = black_sab.replace("\n",".")
	bl = sentencizer(nlp(black_sab))
	bl_sents = [ span.text.strip() for span in bl.sents ]	

with open("Nirvana.txt", encoding="utf-8") as f:
	nirv_text = f.read()
	nirv_text = nirv_text.replace("\n",".")	
	nirv = sentencizer(nlp(nirv_text))
	nirv_sents = [span.text.strip() for span in nirv.sents] 

with open("beatles.txt", encoding="utf-8") as f:
	beatles_text = f.read()
	beatles_text = beatles_text.replace("-",'')
	beatles_text = beatles_text.replace("]",'')
	beatles_text = beatles_text.replace("[",'')
	beatles_text = beatles_text.replace("\n",'.') 
	beatles_text = beatles_text.replace('"','')
	be = sentencizer(nlp(beatles_text))
	be_sents = [ span.text.strip() for span in be.sents ]

max_len = 1000
nirv_sents = nirv_sents[0:max_len]
be_sents = be_sents[0:max_len]
bl_sents = bl_sents[0:max_len]

#print(len(nirv_sents))
#print(len(be_sents))
#print(len(bl_sents))


gold_labels = np.concatenate([
	np.repeat(np.array([1,0,0]).reshape((1,3)), len(nirv_sents), axis=0),
	np.repeat(np.array([0,1,0]).reshape((1,3)), len(be_sents), axis=0),
	np.repeat(np.array([0,0,1]).reshape((1,3)), len(bl_sents), axis=0),
    ])

###
text = (nirv_sents + be_sents + bl_sents)
###
t = Tokenizer()
t.fit_on_texts(text)

vocab_len = len(t.word_index) + 1

E = np.zeros(shape=(vocab_len, 50))
with open("glove.6B.50d.txt", encoding='utf-8') as f:
    for line in f:
        stuff = line.split()
        word = stuff[0]
        numbers = np.array([ float(x) for x in stuff[1:] ])
        if word in t.word_index: 
            E[t.word_index[word],:] = numbers



def eval(vocab_len,num_iterations,gold_labels,maxlen):
	scores = np.empty(num_iterations)
	encoded = t.texts_to_sequences(text)
#	padded = pad_sequences(encoded, maxlen=10, padding="post")
	for ite in range(num_iterations):
		padded = pad_sequences(encoded, maxlen=maxlen, padding="post")
		Xtrain, Xtest, ytrain, ytest = train_test_split(padded, gold_labels,test_size=.2)	
		input_layer = Input(shape=(padded.shape[1],))
		e_layer = Embedding(vocab_len, 50, input_length=maxlen, weights=[E],trainable=False)(input_layer)
		f_layer = Flatten()(e_layer)
		hidden_layer = Dense(20, activation='relu')(f_layer)
		output_layer = Dense(3, activation='sigmoid')(hidden_layer)
		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(Xtrain, ytrain, epochs=25, verbose=0)
		scores[ite] = model.evaluate(Xtest,ytest)[1]
	return scores

n = 7
ites = 20 
test = eval(vocab_len,ites,gold_labels,n)
print(f"Mean: {np.mean(test)}")
print(f"Std Dev: {np.std(test)}")















