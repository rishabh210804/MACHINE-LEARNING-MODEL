# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample SMS dataset (works offline)
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message': [
        "Hello, how are you?",
        "You won 1 lakh rupees! Call now!",
        "Are we meeting today?",
        "Free recharge if you click this link!",
        "See you at the gym",
        "Congratulations, you've won a bike!",
        "I'll call you later",
        "Click to claim your prize now!",
        "Lunch at 2?",
        "Get cash bonus now just by signing up!"
    ]
}

# Load dataset
df = pd.DataFrame(data)

# Preprocessing: Convert label to numbers
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.3, random_state=0)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict using the trained model
y_pred = model.predict(X_test_vec)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
