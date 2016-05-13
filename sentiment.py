from __future__ import division
import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import datetime
from collections import Counter
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def cal_per_in_list(total,fil_count):
    print datetime.datetime.now()
    #filtered = filter(lambda small_list: word in small_list, list)
    #count = len(filtered)
    if (fil_count/total)*100 > 1:
        return True
    else:
        return False    


def cal_per_in_neg(value, total):
    return (value/total)*100

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    #       print stopwords
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    pos_list = []
    for tweet in train_pos: 
        pos_list.append(list(set(tweet)))
    wordList = [item for sublist in pos_list for item in sublist]       
    filteredWords =[w for w in wordList if w not in stopwords]
    wordCount = Counter(filteredWords)
    length = len(train_pos)/100 
    positive = dict((key,value) for key,value in wordCount.items() if value >= length)
    
    neg_list = []
    for tweet in train_neg: 
        neg_list.append(list(set(tweet)))
    wordList = [item for sublist in neg_list for item in sublist]       
    filteredWords =[word for word in wordList if word not in stopwords]
    wordCount = Counter(filteredWords)
    length = len(train_neg)/100 
    negative = dict((key,value) for key,value in wordCount.items() if value >= length)
    features = []   
    for key in positive.keys():
        if key in negative.keys():
            if positive[key] >= 2*negative[key]:
                features.append(key)
        else:
            features.append(key)    
    for key in negative.keys():
        if key in positive.keys():
            if negative[key] >= 2*positive[key]:
                features.append(key)
        else:
            features.append(key)  

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    
    for text in train_pos:
        pos_list = []
        for word in features:   
            if word in text:
                pos_list.append(1)
            else:
                pos_list.append(0)
        train_pos_vec.append(pos_list)

    for text in train_neg:
        neg_list = []
        for word in features:   
            if word in text:
                neg_list.append(1)
            else:
                neg_list.append(0)
        train_neg_vec.append(neg_list)

    for text in test_pos:
        pos_list = []
        for word in features:   
            if word in text:
                pos_list.append(1)
            else:
                pos_list.append(0)
        test_pos_vec.append(pos_list)

    for text in test_neg:
        neg_list = []
        for word in features:   
            if word in text:
                neg_list.append(1)
            else:
                neg_list.append(0)
        test_neg_vec.append(neg_list)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec    


def tweet_to_doc(uid,review,label):
    labeledSent = LabeledSentence(review, tags=[label+'%s' % uid])
    return labeledSent
     


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# Doc2Vec requires LabeledSentence objects as input.
	# Turn the datasets from lists of words to lists of LabeledSentence objects.
	# YOUR CODE HERE

	## Find the word list 

	labeled_train_pos = []
	labeled_train_neg = []
	labeled_test_pos  = []
	labeled_test_neg  = []
	i=0
	for tweet in train_pos:
	 	sentence = tweet_to_doc(i, tweet,'TRAIN_POS_')
	 	i = i+1
	 	labeled_train_pos.append(sentence)
	i=0    
	for tweet in train_neg:
	 	sentence = tweet_to_doc(i, tweet,'TRAIN_NEG_')
	 	i = i+1
	 	labeled_train_neg.append(sentence)
	i=0   
	for tweet in test_pos:
	 	sentence = tweet_to_doc(i, tweet,'TEST_POS_')
	 	i = i+1
	 	labeled_test_pos.append(sentence)
	i=0   
	for tweet in test_neg:
	 	sentence = tweet_to_doc(i, tweet,'TEST_NEG_')
	 	i = i+1
	 	labeled_test_neg.append(sentence)          

	# Initialize model
	model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
	sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
	model.build_vocab(sentences)

	# Train the model
	# This may take a bit to run 
	for i in range(5):
		print "Training iteration %d" % (i)
		random.shuffle(sentences)
		model.train(sentences)

	# Use the docvecs function to extract the feature vectors for the training and test data
	# YOUR CODE HERE
	train_pos_vec = [] 
	train_neg_vec = []
	test_pos_vec = []
	test_neg_vec = []

	for x in range(0, len(labeled_train_pos)):
		s = "TRAIN_POS_"+str(x)
		train_pos_vec.append(model.docvecs[s])

	for x in range(0, len(labeled_train_neg)):
		s = "TRAIN_NEG_"+str(x)
		train_neg_vec.append(model.docvecs[s])

	for x in range(0, len(labeled_test_pos)):
		s = "TEST_POS_"+str(x)
		test_pos_vec.append(model.docvecs[s])

	for x in range(0, len(labeled_test_neg)):
		s = "TEST_NEG_"+str(x)
		test_neg_vec.append(model.docvecs[s])

	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    final_vec = train_pos_vec
    final_vec.extend(train_neg_vec)
    nb_model = BernoulliNB(alpha=1.0, binarize=None, class_prior=None, fit_prior=True)
    nb_model.fit(final_vec, Y)
    lr_model = LogisticRegression()
    lr_model.fit(final_vec, Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
	"""
	Returns a GaussianNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
	# For LogisticRegression, pass no parameters
	# YOUR CODE HERE
	final_vec = train_pos_vec
	final_vec.extend(train_neg_vec)
	#print final_vec
	nb_model = GaussianNB()
	nb_model.fit(final_vec, Y)
	lr_model = LogisticRegression()
	lr_model.fit(final_vec, Y)
	return nb_model, lr_model
   



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    # final_vec = test_pos_vec
    # final_vec.extend(test_neg_vec)
    # confusion_matrix = model.predict(test_pos_vec)
    # pos_count = len(test_pos_vec)
    # neg_count = len(test_neg_vec)
    # pos_list = confusion_matrix[:pos_count]
    # print Counter(pos_list)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    test_pos_vec = numpy.array(test_pos_vec)
    test_neg_vec = numpy.array(test_neg_vec)

    for var in test_pos_vec:
        if model.predict(var.reshape(1, -1)) == ['pos']:
            tp = tp + 1
        else:
            fn = fn + 1
    for var in test_neg_vec:
        if model.predict(var.reshape(1, -1)) == ['neg']:
            tn = tn + 1
        else:
            fp = fp + 1               
    accuracy = (tp+tn)/(tp+tn+fp+fn)        

   
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
