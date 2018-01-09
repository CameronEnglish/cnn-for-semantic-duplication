import nltk
import csv
import string
from nltk.corpus import stopwords

#http://www.nltk.org/


reader = csv.reader(open('/home/cameron/CS/CSTensorFlow/train.csv', 'rU'), delimiter= ",", quotechar='"')
target = open('/home/cameron/CS/CSTensorFlow/output.txt', 'w')
a= 0
stop_words = {'the', 'a', 'and', 'but', 'it', 's', 'd', 'to', 'for', 'of', 'like', 'he', 'she', 'is', 'are', '?', '(', ')', 'i', 'we', '\\', '\'', ',', ',', '`', '\'s', '``', '\'\'', '[', ']', '{', '}', '-', '^', ';', '_', '+', ':'}

for line in reader:
    for field in line:
        a = a+1
        tokens = nltk.word_tokenize(field)
        tokens = [t.lower() for t in tokens]
        cleaned_tokens = filter(lambda x: x not in stop_words, tokens)
        output = (" ".join(map(str, cleaned_tokens)))
        output = output.strip('"`\'')
        output = ''.join([i for i in output if not i.isdigit()])
        output = output.replace("\\","")
        output = output.replace("^","")
        output = output.replace("_","")
        output = output.replace(",","")
        output = output.replace(".","")
        output = output.replace("=","")
        output = output.replace("\"","")
        if a == 6: 
            #output = output + '\n'
            a = 0
        if(len(output) > 3):
            target.write(" " +output +'\n')
            print("writing ", output + '\n')

target.close()



# id q1id q2id q1 q2 is_duplicate
