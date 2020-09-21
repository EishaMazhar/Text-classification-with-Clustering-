# Text Document Classification & Clustering(code of KNN & clustring algorithm from scratch) 

Note: before running the code, kindly check the path 'bbcsports' folder in the directory.

## DataSet: 

### Total Documents: 737, Terms: 4613

The data contain BBC-Sports documents related to 5 different sports:
* Athletics
* Cricket
* Football
* Rugby
* Tennis

## Algorithms Implemented

### Assignment Objective
This assignment focuses on the tasks of classification and clustering.
#### Classification
First task is related to document classification. We have discussed three methods for the task of
text/document classification. These are Rocchio’s, Naïve Bayesian and KNN. In this assignment
you need to work on KNN using VSM model to hold documents and Euclidian distance measure
to estimate closeness of the instances in neighborhood. There is no training required for the KNN
algorithm, all you need to divide your data into suitable splits of train and test. Using the training
data, you need to check the labels of test instances using k=3. From the 5 classes in the dataset you
need to create you train and test sets from 737 instances. Using your idea of a good split set you
are free to create the train and test set. The evaluation of classification will be performed on
Accuracy of classification for the test split. For a detail description of KNN read the chapter 14 of
the textbook.
#### Clustering
For the task of clustering you need to implement a K-means clustering algorithm, assuming all
documents are represented by VSM of suitable feature space. There are 4613 features in the
dataset. The feature selection is what you need to perform as per your understanding. The
evaluation of clustering will be performed by measure purity of the dataset. For a detail description
of K-means read the chapter 16 of the textbook. 

## Output:

### Clustering
<img src='https://github.com/EishaMazhar/Text-classification-with-Clustering-/blob/master/clustering%20output.PNG'/>
