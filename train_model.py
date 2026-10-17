
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nMessage categories:")
print(data["label"].value_counts())

# Convert labels into numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
predictions = model.predict(X_test_vectorized)

print("\nModel training completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Not Spam", "Spam"]
))

# Save the model and vectorizer
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel saved successfully!")
print("Files created: model.pkl and vectorizer.pkl")
