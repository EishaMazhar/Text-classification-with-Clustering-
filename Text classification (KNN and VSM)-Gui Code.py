#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
from nltk.corpus import stopwords
import string
import math
import numpy as np
from nltk.tokenize import RegexpTokenizer
from scipy.spatial import distance
from nltk.stem import PorterStemmer
import random

porter_stemmer=PorterStemmer()
stop_words = set(stopwords.words('english'))


# In[2]:


def getdatafromfile():
    classes=['athletics','cricket','football','rugby','tennis']
    list=[glob.glob('bbcsport/{}/*'.format(classes[i])) for i in range(len(classes))]
    return list,classes


# In[13]:


class TextClassificationKNN:
    def __init__(self,list,classes):
        self.list=list
        self.classes=classes
        self.totalfiles=[len(i) for i in list]
        print("Total files in each class: ",self.totalfiles)
        
        
    def preprocessData(self):

        self.alldocDict={}
        for mainClass in range(len(self.list)): 
            for subFiles in range(len(self.list[mainClass])):
                f=open(self.list[mainClass][subFiles],'r')
                tokenizer = RegexpTokenizer(r'\w+')
                # convert to lower case
                fullfile = tokenizer.tokenize(f.read().lower())
                # stem document
                stemmedDocs=[porter_stemmer.stem(word) for word in fullfile]
                #trimming the file name and removing redundant '.txt'
                p=os.path.basename(self.list[mainClass][subFiles])
                p=p.split('.')[0]
                # remove all tokens that are not alphabetic and stop words
                tokens_without_sw = [word for word in stemmedDocs if word not in stop_words and word.isalpha()]
                self.alldocDict[mainClass,int(p)]=tokens_without_sw

        return self.alldocDict
    
    
    def SplitTrainTestDocs(self,testRatio):    
        
        self.train_count=[len(i)-(int(len(i)*(testRatio))) for i in self.list]
        
        self.Train_documents=[]
        self.Test_documents=[]
        self.Train_set=[]
        
        Train_Init=[]
        for classId in range(len(self.classes)):
            Train_Init.append(random.sample(range(1, self.totalfiles[classId]), self.train_count[classId]))

        for classId in range(len(self.classes)):
            for i in Train_Init[classId]:
                self.Train_documents.append((classId,i))
                self.Train_set.append(self.alldocDict[(classId,i)])

        for class_id in self.alldocDict.keys():
            if(class_id not in self.Train_documents):
                self.Test_documents.append(class_id)

        print("Number of test documents: ",len(self.Test_documents))
        print("Number of train documents: ",len(self.Train_documents))
        
        # index of words in training set(train_set has all the words that are in training set)
        self.word_index={}
        for doc in range(len(self.Train_set)):
            for word in self.Train_set[doc]:
                if word not in self.word_index:
                    self.word_index[word]={}
                    
        print(len(self.word_index))

        return self.Train_documents, self.Test_documents, self.word_index
    
    
    def tfidfCalculation(self,term_index):
        
        f = open("TFIDFdict_allDocs.txt","w")
        idf={}
        for word in term_index.keys():
            df=0
            for doc in self.alldocDict.keys():
                if word in self.alldocDict[doc]:
                    df+=1
            idf[word]=math.log(len(self.Train_documents)/(df))

        for word in term_index.keys():
            for doc in self.alldocDict.keys():      
                if word in self.alldocDict[doc]:
        #             print('word found in ',doc)   
                    term_index[word][doc]=((self.alldocDict[doc].count(word))*idf[word])  #calculate tfidf
                else:
                    term_index[word][doc]=0
        f.write(str(term_index))
        f.close()
        
        return term_index
    
    def formVectors(self,tfidf_dict):

        #form train and test vectors
        self.trainDoc_vect={}
        self.testDoc_vect={}

        f1 = open("TrainDocs_vector.txt","w")
        f2 = open("TestDocs_allDocs.txt","w")

        for docid in self.Train_documents:
            self.trainDoc_vect[docid]=[]
            for word in tfidf_dict.keys():
                self.trainDoc_vect[docid].append(tfidf_dict[word][docid])

        for docid in self.Test_documents:
            self.testDoc_vect[docid]=[]
            for word in tfidf_dict.keys():
                self.testDoc_vect[docid].append(tfidf_dict[word][docid])

        f1.write(str(self.trainDoc_vect))
        f1.close()
        f2.write(str(self.testDoc_vect))
        f2.close()

        return self.trainDoc_vect, self.testDoc_vect
    
    
    def calculateSimilarityAndKNN(self):
        self.testsimilarity={}
        for testid in self.testDoc_vect:
            self.testsimilarity[testid]={}
            for trainid in self.trainDoc_vect:
                self.testsimilarity[testid][trainid]=(distance.cosine(self.testDoc_vect[testid], self.trainDoc_vect[trainid]))
        self.correct=0
        self.no=0
        for i in self.testsimilarity:
            n1,n2,n3=sorted(self.testsimilarity[i], key=self.testsimilarity[i].get)[0:3]
            newList=[n1[0],n2[0],n3[0]]
            frequentNN=max(set(newList), key = newList.count)
        #     print(i[0],frequentNN)
            if(i[0]==frequentNN):
                self.correct+=1
            else:
                self.no+=1
    
    
    def getAccuracy(self):
        return (self.correct/(self.no+self.correct))*100
        


# In[14]:


list,classes=getdatafromfile()
clf=TextClassificationKNN(list,classes)
alldocDict=clf.preprocessData()
Train_documents, Test_documents, word_index=clf.SplitTrainTestDocs(0.30)


# In[15]:


tfidf_dict=clf.tfidfCalculation(word_index)
trainVector,testVector=clf.formVectors(tfidf_dict)


# In[16]:


clf.calculateSimilarityAndKNN()


# In[17]:


#due to random sampling, accuracy will vary between 96-99%

print(clf.getAccuracy())


# In[ ]:





# In[ ]:




