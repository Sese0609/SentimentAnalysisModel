#System should be able to classify positive and negative sentiments
#There will be junk
import nltk
import numpy as np
from nltk.corpus import stopwords, movie_reviews
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('stopwords')
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Positive = 1, Negative = 0
stop_words = set(stopwords.words('english'))

# load the training data
data_train = pd.read_csv('sentiment_analysis_training_data.csv')
# extract the text
text_train = data_train['text'].values
# and the labels
labels_train = data_train['label'].values
data_val = pd.read_csv('sentiment_analysis_validation_data.csv')
# extract the text
text_val = data_val['text'].values
# and the labels
labels_val = data_val['label'].values
#Test data extraction
data_test = pd.read_csv('sentiment_analysis_test_data.csv')
text_test = data_test['text'].values

#Data Visualisation
def print_text(text, label):
  if label == 0:
    print (text, '\nlabel == 0')
  else:
    print (text, '\nlabel == 1')


def get_confusion_matrix(true_label, pred_label):
  """
  Calculate the confusion matrix for your predicted labels. See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
  :param pred_label: Array of predicted labels
  :param true_label: Array of corresponding ground truth (test) labels
  :return: Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
  """
  return confusion_matrix(true_label, pred_label)

def save_as_csv(pred_labels, location='.'):
    """
    Save the labels out as a .csv file
    :pred_labels: numpy array of shape (no_test_labels,) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert pred_labels.shape[0] == 1434, 'wrong number of labels, should be 1434 test labels'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',', fmt='%d')


def preprocessing(text):
    """"
    Cleans raw text, gets rid of punctuation, numbers, stopwords and lowercases everything.
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            clean_tokens.append(word)

    cleaned_text = ' '.join(clean_tokens)
    return cleaned_text

#Applies the preprocessing function to every review
clean_text_train = [preprocessing(review) for review in text_train]
print("Cleaning validation data...")
clean_text_val = [preprocessing(review) for review in text_val]


#TfIDF used to extract numerical features from text, matrix is limited to 5000 max features.
print("Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(clean_text_train)
X_val_features = vectorizer.transform(clean_text_val)
print(f"Feature extraction complete! Training matrix shape: {X_train_features.shape}")

#IsolationForest detects anomolies
spam_detector = IsolationForest(contamination=0.1, random_state=42)
spam_predictions = spam_detector.fit_predict(X_train_features)
clean_X_train = X_train_features[spam_predictions == 1]
clean_labels_train = labels_train[spam_predictions == 1]
spam_removed = len(labels_train) - len(clean_labels_train)
print(f"Successfully removed {spam_removed} spam emails!")
print(f"The new, clean training data shape is: {clean_X_train.shape}")


#Sentiment Classifier - Detects the emotional tone.
print("Now training the sentiment classifier...")
sentiment_classifier = LogisticRegression(max_iter=1000, random_state=42)
sentiment_classifier.fit(clean_X_train, clean_labels_train)
print("Training complete now, the model is now able to make predictions!")

#ConfusionMatrix for validation data
print('Now evaluating the validation data...')
val_predictions = sentiment_classifier.predict(X_val_features)
conf_matrix = get_confusion_matrix(labels_val, val_predictions)
tn, fp, fn, tp = conf_matrix.ravel()
print(conf_matrix)
print(f"True Negatives  (Correctly labeled Negative)  : {tn}")
print(f"False Positives (Incorrectly labeled Positive): {fp}")
print(f"False Negatives (Incorrectly labeled Negative): {fn}")
print(f"True Positives  (Correctly labeled Positive)  : {tp}")
val_correct = np.trace(conf_matrix)
val_total = np.sum(conf_matrix)
val_accuracy = (val_correct / val_total) * 100
print(f"Overall Accuracy on Validation dataset: {val_accuracy:.2f}%")
val_precision = (tp / (tp + fp)) * 100
val_recall = (tp / (tp + fn)) * 100
print(f"Precision: {val_precision:.2f}%")
print(f"Recall   : {val_recall:.2f}%")
print('now pulling 4 examples of sentences: ')
tp_index = np.where((labels_val == 1) & (val_predictions == 1))[0][0]
print("\nTRUE POSITIVE (A Win: Guessed Good, was actually Good)")
print(f"Text: '{text_val[tp_index]}'")
tn_index = np.where((labels_val == 0) & (val_predictions == 0))[0][0]

print("\nTRUE NEGATIVE (A Win: Guessed Bad, was actually Bad)")
print(f"Text: '{text_val[tn_index]}'")

fp_index = np.where((labels_val == 0) & (val_predictions == 1))[0][0]
print("\nFALSE POSITIVE (Failure Case: Guessed Good, was actually Bad)")
print(f"Text: '{text_val[fp_index]}'")

fn_index = np.where((labels_val == 1) & (val_predictions == 0))[0][0]
print("\nFALSE NEGATIVE (Failure Case: Guessed Bad, was actually Good)")
print(f"Text: '{text_val[fn_index]}'")

#Vectorizes test data, detects spam and predicts test labels.
clean_text_test = [preprocessing(review) for review in text_test]
x_test_features = vectorizer.transform(clean_text_test)
print('\nDetecting spam within test data')
test_spam_predictions = spam_detector.predict(x_test_features)
final_test_predictions = sentiment_classifier.predict(x_test_features)
final_test_predictions[test_spam_predictions == -1] = -1
print(f"Spam detected in test set: {np.sum(test_spam_predictions == -1)}")

unique_labels, counts = np.unique(final_test_predictions, return_counts=True)
for label, count in zip(unique_labels, counts):
    if label == -1:
        print(f"Spam/Anomalies (-1): {count}")
    elif label == 0:
        print(f"Negative (0)       : {count}")
    elif label == 1:
        print(f"Positive (1)       : {count}")

save_as_csv(final_test_predictions)



#External Dataset Evaluation
print('\nNow starting NLTK movie_reviews evaluation')
nltk_texts = []
nltk_labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        nltk_texts.append(movie_reviews.raw(fileid))
        nltk_labels.append(1 if category == 'pos' else 0 )

print(f'Loaded {len(nltk_texts)} reviews from movie_reviews.')

#Preprocessing
print("Cleaning NLTK data (may take a few seconds)...")
clean_nltk_texts = [preprocessing(review) for review in nltk_texts]

#Feature extraction with already trained TF-IDF
print("Extracting features using existing TF-IDF vocabulary...")
nltk_features = vectorizer.transform(clean_nltk_texts)

#Predict and Evaluate
print("\nPredicting sentiment on NLTK data...")
nltk_predictions = sentiment_classifier.predict(nltk_features)

#Calculate and print the confusion matrix
nltk_conf_matrix = get_confusion_matrix(nltk_labels, nltk_predictions)
print("Confusion Matrix for NLTK movie_reviews:")
print(nltk_conf_matrix)
tn, fp, fn, tp = nltk_conf_matrix.ravel()
print(f"True Negatives  (Correctly labeled Negative)  : {tn}")
print(f"False Positives (Incorrectly labeled Positive): {fp}")
print(f"False Negatives (Incorrectly labeled Negative): {fn}")
print(f"True Positives  (Correctly labeled Positive)  : {tp}")

correct_predictions = np.trace(nltk_conf_matrix)
total_predictions = np.sum(nltk_conf_matrix)
accuracy = (correct_predictions / total_predictions) * 100
print(f"Overall Accuracy on NLTK dataset: {accuracy:.2f}%")
nltk_precision = (tp / (tp + fp)) * 100
nltk_recall = (tp / (tp + fn)) * 100
print(f"Precision: {nltk_precision:.2f}%")
print(f"Recall   : {nltk_recall:.2f}%")
