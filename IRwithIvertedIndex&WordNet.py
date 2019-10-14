import numpy as np
import operator
import math
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
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
a= []
b = []
c = []
new_filt = []
COS = {}
COS2 = {}
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
matrix_q = np.zeros((T,))
matrix_wn = np.zeros((T,))
q_term = X[-1]
q_length = len(q_term)
for word in q_term:
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            a.append(lemma.name())
    dog = wn.synsets(word)
    tod = dog[0].hypernyms()
    types_of_dog = dog[0].hyponyms()
    for synse in tod:
        for lemma in synse.lemmas():
            b.append(lemma.name())
    for synse in types_of_dog:
        for lemma in synse.lemmas():
            c.append(lemma.name())
    a.extend(b)
    a.extend(c)
new_a = []
for j in a:
    j = j.lower()
    if j not in stop_words and j not in punct_list:
        j = (lemmatizer.lemmatize(j))
        new_a.append(j)
a = new_a
for word in q_term:
        tf_q=0
        for word_2 in q_term:
            if word == word_2:
                tf_q+=1
        tf_q = tf_q/math.sqrt(tf_q)
        tf_q = tf/q_length
        if tf_q > 0.001:
            matrix_q[vocab[word]]=tf_q
        else:
            matrix_q[vocab[word]]=0
#-----------------------------Query-Processing-------------------------------#
for word in a:
        tf_q=0
        for word_2 in a:
            if word == word_2:
                tf_q+=1
        tf_q=tf_q/math.sqrt(tf_q)
        tf_q = tf/len(a)
        if word in vocab.keys():
            if tf_q > 0.0001:
                matrix_wn[vocab[word]]=tf_q*IDF[word]
            else:
                matrix_wn[vocab[word]]=0
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
for term in a:
    if term not in q_term:
        if term in Inverted_Index.keys():
            new_filt.extend(Inverted_Index[term])
#------------------------Cosine-Similarity-----------------------------------#
for index in filt_doc:
    cos = float(np.dot(matrix_IDF[index-1],matrix_q))/(amp(matrix_IDF[index-1])*amp(matrix_q))
    COS[index] = cos
for index_1 in new_filt:
    cos2 = float(np.dot(matrix_IDF[index_1-1],matrix_wn))/(amp(matrix_IDF[index_1-1])*amp(matrix_wn))
    COS2[index_1] = cos2
sorted_COS = sorted(COS.items() , key = operator.itemgetter(1) , reverse=True)
sorted_COS2 = sorted(COS2.items() , key = operator.itemgetter(1) , reverse=True)
print("Search Complete. Results Fetched!")
print("Documents matching Query term:> ")
print(sorted_COS)
print("\n")
print("Documents matching Synsets of query term:> ")
print(sorted_COS2)
#------------------------Cosine-Similarity-----------------------------------#
