import nltk
import numpy as np
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


train_file = open("trainWithoutDev.txt", "r")
dev_file = open("dev.txt", "r")

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform("trainWithoutDev.txt")
#X_train_counts.shape
#
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tfidf.shape
#
#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

def remove_stop_words(line): 
    stop_words = set(stopwords.words('english'))
    
    word_tokens = word_tokenize(line)
    
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words and w not in (',', '&', '(', ')', '.'):
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)

def stem(line):
    stemmer = PorterStemmer()
    singles = []
    for plural in line.split(" "):
        singles.append(stemmer.stem(plural))
    return ' '.join(singles)

def parse(f):
    final_list = []
    for line in f:
        line_list = line.split("\t")
        if (len(line_list) > 3):
            line_list.pop(1)
        line_list[1] = stem(line_list[1])
        line_list[2] = stem(line_list[2])  
        line_list[1] = remove_stop_words(line_list[1])
        line_list[2] = remove_stop_words(line_list[2]) 
        final_list.append(line_list)
        if (len(line_list) != 3 ):
          print('oops');
          print(line_list)
    return final_list

train_list = parse(train_file)
test_list = parse(dev_file)

print(train_list)

train_set = []
test_set = []
for l in train_list:
    train_set.append((dict(question = l[1], answer = l[2], question_len = len(l[1]), answer_len = len(l[2])), l[0]));

for l in test_list:
    test_set.append((dict(question = l[1], answer = l[2], question_length = len(l[1]),  answer_len = len(l[2])), l[0]));

print(train_set[0])

#classifier = nltk.NaiveBayesClassifier.train(train_set)
#classifier2 = SklearnClassifier(BernoulliNB()).train(train_set)
#classifier3 = nltk.DecisionTreeClassifier.train(train_set)
#classifier4 = SklearnClassifier(LinearSVC()).train(train_set)
#print("Naive Bayes: " + str(nltk.classify.accuracy(classifier, test_set)))
#print("Bernoulli " + str(nltk.classify.accuracy(classifier2, test_set)))
#print("Decision tree: " + str(nltk.classify.accuracy(classifier3, test_set)))
#print("Train set accuracy: " + str(nltk.classify.accuracy(classifier4, train_set)))
#print("Linear: " + str(nltk.classify.accuracy(classifier4, test_set)))

train_file.close()
dev_file.close()


