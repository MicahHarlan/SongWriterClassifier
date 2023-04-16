from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense
from keras.models import Model
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
	nirv_text = nirv_text.replace("[","")
	nirv_text = nirv_text.replace("]","")	 
	nirv_text = nirv_text.replace("',",".")	
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

#Switch board
mode= 'tfidf'


####

gold_labels = np.concatenate([
	np.repeat(np.array([1,0,0]).reshape((1,3)), len(nirv_sents), axis=0),
	np.repeat(np.array([0,1,0]).reshape((1,3)), len(be_sents), axis=0),
	np.repeat(np.array([0,0,1]).reshape((1,3)), len(bl_sents), axis=0),
    ])

t = Tokenizer()
t.fit_on_texts(nirv_sents + be_sents + bl_sents)

dtm = t.texts_to_matrix(nirv_sents + be_sents + bl_sents,mode=mode)

input_layer = Input(shape=(dtm.shape[1],))
hidden_layer = Dense(13, activation='relu')(input_layer)
output_layer = Dense(3, activation='softmax')(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dtm, gold_labels, epochs=30, verbose=0)


sent = input("Enter a sentence: ")
while (sent != 'done'):
	coded = t.texts_to_matrix([ sent ], mode=mode)
	results = (model.predict(coded))
	print(f"Nirvana: {results[0][0]:.2f}")	
	print(f"The Beatles: {results[0][1]:.2f}")
	print(f"Black Sabbath:  {results[0][2]:.2f}")
	sent = input("Enter a sentence: ") 









