from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
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
#	nirv_text = nirv_text.replace("[","")
#	nirv_text = nirv_text.replace("]","")	 
#	nirv_text = nirv_text.replace("',",".")	
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

#Switch board
mode= 'count'


####

gold_labels = np.concatenate([
	np.repeat(np.array([1,0,0]).reshape((1,3)), len(nirv_sents), axis=0),
	np.repeat(np.array([0,1,0]).reshape((1,3)), len(be_sents), axis=0),
	np.repeat(np.array([0,0,1]).reshape((1,3)), len(bl_sents), axis=0),
    ])

t = Tokenizer()
t.fit_on_texts(nirv_sents + be_sents + bl_sents)

def eval(sents,num_iterations,gold_labels,epochs,mode):
	dtm = t.texts_to_matrix(nirv_sents + be_sents + bl_sents,mode=mode)
	scores = np.empty(num_iterations)	
	for ite in range(num_iterations):
		Xtrain, Xtest, ytrain, ytest = train_test_split(dtm, gold_labels,test_size=.2)	
		input_layer = Input(shape=(dtm.shape[1],))
		hidden_layer = Dense(20, activation='relu')(input_layer)
		output_layer = Dense(3, activation='softmax')(hidden_layer)
		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(Xtrain,ytrain, epochs=epochs, verbose=0)
		scores[ite] = model.evaluate(Xtest, ytest)[1]
	return scores

iterations = 20
epochs = 23
mode = ['binary','count','freq','tfidf']
#for m in mode:
test_model = eval(nirv_sents+be_sents+bl_sents,iterations,gold_labels,epochs,mode[0])
print(f"Mean: {np.mean(test_model)}")
print(f"Std Dev: {np.std(test_model)}")










