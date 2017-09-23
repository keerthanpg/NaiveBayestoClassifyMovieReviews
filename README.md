# NaiveBayestoClassifyMovieReviews

A Naive Bayes Classifier, written from scratch, without using dedicated ML libraries like scikit-learn, to classify movie reviews from IMDB as positive or negative. This is part of my submission for 10-601B, Instroduction to Machine Learning at CMU. 

After cloning or downloading and extracting, run hw_script.py to train/test or modify the classier. 

## Guide to Variables

1. vocabulary is a V × 1 dimensional vector that contains every word, excluding stop words, appearing in the passages. 
2. stop words is a S × 1 dimensional vector that contains every stop word appearing in the passages.
3. XTrain is a n × V dimensional matrix describing the n documents used for training the Naıve Bayes classifier. The entry XTrain(i,j) is 1 if word j appears in the i-th training passage and 0 otherwise.
4. yTrain is a n×1 dimensional vector containing the class labels for the training documents. yTrain(i) is 0 if the i-th passage is a negative review and 1 if it is a positive one.
5. XTest and yTest are the same as XTrain and yTrain, except instead of having n rows, they have m rows. 
6. XTrainSmall and yTrainSmall are subsets of XTrain and yTrain.
