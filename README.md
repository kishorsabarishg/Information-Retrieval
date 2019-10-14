# Information-Retrieval
In this, the documents pertaining to the categories "alt.atheism" and "comp.graphics" of the 20NEWSGROUPS dataset are fetched.
Here the user enters a query term as input if the query term is present in the vocabulary, the documents are fetched.
IRwithInvertedIndex.py is an IR system without WordNet.
IRwithoutInvertedIndex.py is an IR system without InvertedIndex
IRwithInvertedIndex&WordNet.py is IRwithInvertedIndex.py implementing WordNet.In this code, documents which contain terms which are synonyms,hyponyms and hypernyms to query term are fetched
This is done with the help of tf-idf vectorizer which has implemented directly without using tf-idf vectorizer from sckit learn
The documents are fetched and ranked in descending order of their cosine-similarity with the Query term. The application of Inverted Index is much appreciated while comparing the spped of the former and the latter codes
