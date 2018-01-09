import gensim
from gensim.models import Word2Vec
import glob
import os


#Persist a model to disk with:
#model.save(fname)
#model = Word2Vec.load(fname)  # you can continue training with the loaded model!



# word2vec expects a list of strings as its input it sentences = [['first', 'sentence'], ['second', 'sentence']]
# model = Word2Vec(sentences)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print("working on file ", fname)
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
#sentences = open('/home/cameron/CS/CSTensorFlow/output/output.txt')
sentences = MySentences('/home/cameron/CS/CSTensorFlow/output') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=5, size=200)

model.save('/home/cameron/CS/CSTensorFlow/mymodel')
