# IMDB/Twitter Sentiment Analytics using Supervised Learning:

Sentiment analysis over IMDB-movie reviews and Twitter data using Logistic regression and Naive Bayes' classifiers from Python’s machine learning package: scikit-learn.

### Objective:
* Perform sentiment analysis over IMDB movie review data and Twitter data using supervised learning techniques, Logistic    regression and Naive Bayes Classifiers.
* The data set contained 50,000 movie reviews from IMDB and around 900 K tweets. 
* The raw data were filtered using NLP techniques and converted in to features and then converted into feature vectors.
* Later on, the words list were labelled using Doc2Vec technique.
* The model was trained and tested using Bayes Naive Classifiers and Logistic regression.
* The accuracy of the model was 84%.

### Requirements:
* gensim
* scipy

### Algorithms:
* Naive Bayes Classifier
* Logistic Regression

### Steps to Run :
```python
python sentiment.py
```
* Loads the train and test set into four different lists(Train/Test positive/negative).
* Determine a list of words that will be used as features. 
    * This list should have the following properties:
    *   (1) Contains no stop words
    *   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    *   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
* Turn the datasets from lists of words to lists of LabeledSentence objects.
* Use the docvecs function to extract the feature vectors for the training and test data.
* Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
* For BernoulliNB, use alpha=1.0 and binarize=None
* For LogisticRegression, pass no parameters
* Use the predict function and calculate the true/false positives and true/false negative.

### Output:
![alt tag](https://github.com/sudhansusingh22/IMDB-Twitter-Sentiment-Analytics/blob/master/naivebayes.PNG)
