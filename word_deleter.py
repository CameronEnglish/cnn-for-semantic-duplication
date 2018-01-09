import gensim
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
#once the model has been created, removes words not in the model

model = Word2Vec.load('/home/cameron/CS/CSTensorFlow/mymodel')

fout = open('/home/cameron/CS/CSTensorFlow/output/newOutput.txt', 'w')

with open('/home/cameron/CS/CSTensorFlow/output/output.txt') as f:
    for line in f:
        temp_sentence = ''
        for word in line.split():
            try:
                model.wv[word]
            except Exception:
                #line = line.replace(word, "")
                temp_sentence += ''
            else:
                temp_sentence += word + ' '
        fout.write(temp_sentence + '\n')
        print(temp_sentence + '\n')
