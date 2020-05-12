import glob
import os

emails, labels = [], []
file_path = 'enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
	with open(filename, 'r', encoding = "ISO-8859-1") as infile:
		emails.append(infile.read())
		labels.append(1)

file_path = 'enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
	with open(filename, 'r', encoding = "ISO-8859-1") as infile:
		emails.append(infile.read())
		labels.append(0)

# Clean the data
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(astr):
	return astr.isalpha()

all_names =  set(names.words())
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(docs):
    cleaned_text = []
    for doc in docs:
        cleaned_text.append(' '.join([lemmatizer.lemmatize(word.lower()) for 
                                      word in doc.split() if letters_only(word) 
                                      and word not in all_names]))
    return cleaned_text

cleaned_emails = clean_text(emails)

#Data extraction: removing the stopping words(unuseful)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', max_features = 100000)


#Divide the dataset into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, 
                                                    test_size = 0.33,
                                                    random_state = 0)
term_doc_train = cv.fit_transform(X_train)


#Make the classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = 1.0, fit_prior = True)
clf.fit(term_doc_train, y_train)


# Predict the test results
term_doc_test = cv.transform(X_test)
prediction_proba = clf.predict_proba(term_doc_test)

prediction = clf.predict(term_doc_test)
print(prediction[:10])

#Calculate the accuracy

accuracy = clf.score(term_doc_test, y_test)
print("The accuracy is: {0: .1f}".format(accuracy * 100))

