# SMSspamdetectionexample
I'm currently in the process of learning Scikit-Learn, I'm documenting what I've learned using online resources and websites that teach these concepts. To start of learning Scikit-Learn I am taking a project based approach of learning by doing. I'm using the GitHub repo build-your-own-x by codecrafters-io.

## Resources

- https://kopilov-vlad.medium.com/detect-sms-spam-in-kaggle-with-scikit-learn-5f6afa7a3ca2

* https://scikit-learn.org/stable/supervised_learning.html

* https://github.com/codecrafters-io/build-your-own-x
* https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


## Brief Overview
For the process of spam detection for SMS messages we will use machine learning (ML) to detect if a text message is spam or not spam (called "ham"). The project I am referencing used this data set on Kaggle for this purpose. This dataset has 
a file with 5,574 text messages. Each message is labeled as either ham or spam. Here is a picture of what the data set looks like. 

<img width="1010" alt="image" src="https://github.com/user-attachments/assets/9d64be93-2d23-47fe-b057-e5f485fc6481" />

The data set has to be split into two parts:
* Teach set - To train the computer on what spam and ham look like.
* Test set - To check if the computer can correctly identify spam/ham.

One of the six functionalities of Scikit-Learn is Classification, which we will use this task. 
The library has a wide range of tools for classification, and we will be using classifiers and vectorizers for this task. 
Classifiers are algorithms that categorize data, such as identifying whether a text message is spam or not. 
Examples include decision trees, Naive Bayes, and k-nearest neighbors. These classifiers learn patterns from a dataset (teach-set) and predict outcomes for new data (test-set). 
However, classifiers cannot directly understand plain text, which is where vectorizers come in. 
Vectorizers, such as CountVectorizer and TfidfVectorizer, convert text into numerical representations by analyzing word occurrences and importance. 
CountVectorizer simply counts word frequencies, while TfidfVectorizer adjusts for common words by emphasizing unique ones. 
Together, vectorizers transform text into input data for classifiers, enabling accurate predictions.

### Classification tools

1. Naive Bayes:
* BernoulliNB(): A probabilistic classifier often used for text classification.

2. Tree-Based Models:
* RandomForestClassifier(): Combines many decision trees to make better predictions.
  
* DecisionTreeClassifier(): A single tree that makes decisions by asking yes/no questions.
  
* ExtraTreesClassifier(): Similar to RandomForest but uses slightly different rules for building trees.
  
3. Ensemble Methods:
   
  - AdaBoostClassifier(): Combines weak classifiers to create a strong one. Adds small models together to make one strong mode
  
  - BaggingClassifier(): Averages predictions of multiple base estimators. Combines results from multiple models to improve accuracy.
  
  - GradientBoostingClassifier(): Builds models sequentially to correct previous errors. Builds models step-by-step, fixing mistakes from previous steps.


4. Linear Models:

* PassiveAggressiveClassifier(): Effective for large-scale and sparse data.
  
* RidgeClassifier() and RidgeClassifierCV(): Linear classifiers with regularization. Similar to drawing a straight line but adds some smoothing to avoid overfitting.
  
* SGDClassifier(): Uses stochastic gradient descent for optimization. Uses a fast, step-by-step approach to find the best line for classification.

5. Support Vector Machines:

* OneVsRestClassifier(SVC(kernel='linear')): Applies linear support vector classification in a one-vs-rest approach. Draws boundaries to separate different categories, focusing on one label at a time.

6. Logistic Regression: 

* OneVsRestClassifier(LogisticRegression()): Uses logistic regression in a one-vs-rest approach. A method that predicts the probability of an item belonging to a category, one category at a time.
  
7. Other Methods:

* KNeighborsClassifier(): A distance-based method that classifies based on nearest neighbors. Looks at the nearest examples (neighbors) to classify something.
  
* CalibratedClassifierCV(): Calibrates probabilities for a given classifier. Fine-tunes predictions to give better probabilities
  
* DummyClassifier(): A baseline classifier for comparison. A simple model that makes predictions randomly or using basic rules, just for comparison.

### Text Preprocessing Tools (Vectorizers)
1. CountVectorizer:

* It counts how many times each word appears in the text. For example, if you have sentences like "I like cats" and "I like dogs," it creates a table of word counts for each sentence.
  
2. TfidfVectorizer:

* It also counts words but gives more importance to words that are unique and less common. For example, in a document full of "the" and "is," these common words will have less importance compared to unique words like "cats" or "dogs."
  
3. HashingVectorizer:

* It uses a technique called "hashing" to quickly convert text into numbers. This is faster and saves memory but doesn't keep track of the actual words.

## Data Preparation and Method
The kaggle data set has a file called spam.csv, to divide the data set into teach and test sets we will divide them in the 70:30 ratio. With this we need to make a python code that runs all the classifiers and vectorizers together in combination. This will yield a value for each classifier and vectorizer. The code is enclosed in the scamdetection.ipynb file. With the results I made a table or a CSV file Book1.csv to record each value. Then I need to sort all the 
values to get the maximum value. The Calibrated ClassifierCV with TfidVectorizer yeilded the maximum value, 0.987440191387559. 

<img width="434" alt="image" src="https://github.com/user-attachments/assets/f768029b-ec95-43a0-a3b1-4d8b266eef78" />

Now that we know what classifier and vectorizer works the best, we use this to predict if a message is spam or not using our model. The code is enclosed in prediction.ipynb. After we run the code test_scores.csv is produced. Open the file, do an advanced word search for the word "wrong" and count the number of wrongs. 
We are out of 1673 predictions, 22 are wrong and the rest are right. Which shows a high success rate (98.7%) for our method.

## Deployment
Because I was using Google Colab to run my python files, i had to use ngrok to deploy the API. I ran the script given in APIspam.ipynb and using ngrok I can test if a message is spam or not using the ngrok link that is given after running the script. The set up for ngrok was quite easy and required minimal effort. As shown in the pictures down below.
![image](https://github.com/user-attachments/assets/503a5c44-9405-4b7d-b74a-2f2590e25997)

![image](https://github.com/user-attachments/assets/a58aa536-aa73-4397-8ec9-3d26f1112890)

## Reflection 
This was a good task to take up to learn classification using Scikit-Learn. Can't wait to learn and document more projects like these. 













