import argparse
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import re
from nltk.tokenize import WhitespaceTokenizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords


import nltk
nltk.download('wordnet')

# PARSE INPUT #
parser = argparse.ArgumentParser()
parser.add_argument("-test", dest = "testfilename", default = "dev.txt")
parser.add_argument("-train", dest = "trainfilename", default = "trainWithoutDev.txt")

args = parser.parse_args()

train_file = open(args.trainfilename, "r")
dev_file = open(args.testfilename, "r")

# PARSE TEXT #

def parse(f):
    final_list = []
    for line in f:
        line_list = line.split("\t")
        if (len(line_list) > 3):
            line_list.pop(3) # for dev clean
        line_list[1] = re.sub('\d', '0', line_list[1])
        line_list[2] = re.sub('\d', '0', line_list[2])

        #line_list[1] = re.sub('\(', '', line_list[1])
        #line_list[2] = re.sub('\(', '', line_list[2])
#
        #line_list[1] = re.sub('\)', '', line_list[1])
        #line_list[2] = re.sub('\)', '', line_list[2])

        final_list.append(line_list)
        if (len(line_list) != 3 ):
          print('oops');
          print(line_list)
    return final_list

train_list = parse(train_file)
test_list = parse(dev_file)

# PROCESSING #

def tokenize(x):
    tokenizer = WhitespaceTokenizer()
    word_tokens = tokenizer.tokenize(x)
    stop_words = set(stopwords.words('english'))

    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence

def stemmer(x):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in x])
 
def lemmatize(x):
 lemmatizer = WordNetLemmatizer()
 return ' '.join([lemmatizer.lemmatize(word) for word in x])


# CREATE DB #
df_train = pd.DataFrame(train_list, columns = ['Class', 'Question', 'Answer'])
df_test = pd.DataFrame(test_list, columns = ['Class', 'Question', 'Answer'])

df_train['Question'] = df_train['Question'] + " " + df_train['Answer'] 
df_test['Question'] = df_test['Question'] + " " +  df_test['Answer'] 



df_train['q_tokens'] = df_train['Question'].map(tokenize)
df_train['q_lemma'] = df_train['q_tokens'].map(lemmatize)
df_train['q_stems'] = df_train['q_tokens'].map(stemmer)

df_test['q_tokens'] = df_test['Question'].map(tokenize)
df_test['q_lemma'] = df_test['q_tokens'].map(lemmatize)
df_test['q_stems'] = df_test['q_tokens'].map(stemmer)

X_train = df_train['q_stems']
y_train = df_train['Class']
X_test = df_test['q_stems']
y_test = df_test['Class']


vectorizer = TfidfVectorizer(min_df=1, max_df = 1.0, smooth_idf = False, sublinear_tf=True, use_idf=True)
train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
test_corpus_tf_idf = vectorizer.transform(X_test)



model1 = LinearSVC()

#model2 = MultinomialNB()    
model1.fit(train_corpus_tf_idf,y_train)
#model2.fit(train_corpus_tf_idf,y_train)
result1 = model1.predict(test_corpus_tf_idf)
#result2 = model2.predict(test_corpus_tf_idf)
#print(result1)
#print(result2)
print(accuracy_score(y_test, result1))
#print(accuracy_score(y_test, result2))

##code based on https://medium.com/@ishan16.d/text-classification-in-python-with-scikit-learn-and-nltk-891aa2d0ac4b
##and on https://www.py4u.net/discuss/139790