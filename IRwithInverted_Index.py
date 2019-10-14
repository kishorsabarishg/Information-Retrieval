import numpy as np
import operator
import math
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
newsgroups=fetch_20newsgroups(subset='train',categories=
                              ['alt.atheism','comp.graphics'])
def reemovNestings(l): 
    for i in l: 
        if type(i) == list: 
            reemovNestings(i) 
        else: 
            output.append(i)
X = newsgroups.data
X_temp = X
query = input("Enter the Query term:: ")
str(query)
print("Searching...")
query.lower()
X.append(query)
lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words("english"))
new_X = []
output = []
filt_doc = []
COS = {}
d = []
N = len(X)
punct_list = ["\n",",","(",")","-","[","]","*","{","}","#","@","!",":",";",".","?","|",">","<","+","*","&","'``'","--","_","^","%","/","\\","?","'",'"',"=","~"]
for ngs in X:
    ngs = ngs.lower()
    for charac in punct_list:
        ngs = ngs.replace(charac,' ') 
    ngs = word_tokenize(ngs)
    temp = []
    for word in ngs:
        if word not in stop_words:
            temp.append(lemmatizer.lemmatize(word))
    new_X.append(temp)
X = new_X
reemovNestings(new_X)
temp_X = set()

#----------------------------Vocabulary Buiding------------------------------#

for k in output:
    temp_X.add(k)
vocab = dict.fromkeys(temp_X,1)
count = dict.fromkeys(temp_X,0)
i = 1
for j in list(vocab.keys()):
    vocab[j] = i
    i+=1
#----------------------------Vocabulary Buiding------------------------------#
#for word in list(count.keys()):
#    for doc in X:
#        if word in doc:
#            count[word]+=1
    
    
#-------------------Count and Inverted Index Building------------------------#
    
Inverted_Index = {new_list: set() for new_list in set(vocab.keys())}
doc_id = 1
for doc in X:    
    for word in doc:
        st = 1
        k = doc.count(word)
        if k > 1:
            st = 0
        if st == 1:
            count[word] += 1
        else:
            count[word] += 1/k
        math.ceil(count[word])
        Inverted_Index[word].add(doc_id)
    doc_id += 1

#-------------------Count and Inverted Index Building------------------------#

for word in list(Inverted_Index.keys()):
    Inverted_Index[word] = list(Inverted_Index[word])
T = len(vocab)
T += 1
#-----------------------------------IDF--------------------------------------#
IDF = {}
matrix_IDF = np.zeros((len(X),T))
for words in list(vocab.keys()):
    IDF[words] = math.log((N/count[words]),2)
#matrix_IDF=np.ndarray((len(X),T))
#-----------------------------------IDF--------------------------------------#



#-----------------------------------TF-IDF-----------------------------------#
doc_count = 0
for doc in X:
    n = len(doc)
    for word in doc:
        tf = 0
        for word_2 in doc:
            if word == word_2:
                tf += 1
        tf = tf/math.sqrt(tf)
        tf = tf/n
        matrix_IDF[doc_count][vocab[word]] = tf
        matrix_IDF[doc_count][vocab[word]] *= IDF[word]
    doc_count += 1
#-----------------------------------TF-IDF-----------------------------------#
    
    
#-----------------------------Query-Processing-------------------------------#
matrix_q=np.zeros((T,))
q_term = X[-1]
q_length = len(q_term)
for word in q_term:
        tf_q=0
        for word_2 in q_term:
            if word == word_2:
                tf_q+=1
        tf_q = tf_q/math.sqrt(tf_q)
        tf_q = tf/n
        if tf_q > 0.001:
            matrix_q[vocab[word]]=tf_q
        else:
            matrix_q[vocab[word]]=0
#-----------------------------Query-Processing-------------------------------#


#----------------------------Amplitude---------------------------------------#
def amp(x):
    s=0
    for i in x:
        if(i<0):
           print(i)
        s+=i**2
    return math.sqrt(s)
#----------------------------Amplitude---------------------------------------#
for word in list(Inverted_Index.keys()):
    for i in range(q_length):
        if word == q_term[i]:
            filt_doc.extend(Inverted_Index[word])
#------------------------Cosine-Similarity-----------------------------------#
for index in filt_doc:
    cos = float(np.dot(matrix_IDF[index-1],matrix_q))/(amp(matrix_IDF[index-1])*amp(matrix_q))
    d.append((X_temp[index-1],cos))
    COS[index] = cos
#for index in filt_doc:
#    COS = X_temp[index-1]
sorted_COS = sorted(COS.items() , key = operator.itemgetter(1) , reverse=True)
sorted_d = sorted(d, key=operator.itemgetter(1),reverse=True)
d = sorted_d
print("Search Complete. Results Fetched!")
print(sorted_COS)
#------------------------Cosine-Similarity-----------------------------------#
