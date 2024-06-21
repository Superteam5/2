import requests
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Fetching text from URL
url = "https://gutenberg.ca/ebooks/hemingwaye-oldmanandthesea/hemingwaye-oldmanandthesea-00-t.txt"
response = requests.get(url)
text = response.text

# Preprocess text using NLTK to remove stopwords
stop_words = set(stopwords.words('english'))
words = nltk.word_tokenize(text.lower())
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Word frequency calculation
word_freq = Counter(filtered_words)
most_common_words = word_freq.most_common(10)

# Visualize the top 10 words
top_words_df = pd.DataFrame(most_common_words, columns=['word', 'frequency'])
plt.figure(figsize=(10, 5))
plt.bar(top_words_df['word'], top_words_df['frequency'])
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 10 Words in The Old Man and the Sea')
plt.show()

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Tokenization and NLP processing with spaCy
doc = nlp(text)

# Extracting sentences and calculating sentence scores based on word frequency
sentence_scores = {}
for sent in doc.sents:
    for word in sent:
        if word.text.lower() in word_freq:
            if sent in sentence_scores:
                sentence_scores[sent] += word_freq[word.text.lower()]
            else:
                sentence_scores[sent] = word_freq[word.text.lower()]

# Selecting top sentences for the summary
import heapq
summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
summary = ' '.join([sent.text for sent in summary_sentences])

# Display the summary
print("Summary:")
print(summary)

# Preparing data for machine learning (this is just a placeholder, as summarization does not directly lend itself to ML in this context)
# For illustration, let's consider predicting the frequency of words using a simple regression task
# Generating features and target variable
X = top_words_df[['word']]
y = top_words_df['frequency']

# Convert words to categorical codes
X['word_code'] = X['word'].astype('category').cat.codes

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X['word_code'].values.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Using a Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)# Visualize the top 10 words
top_words_df = pd.DataFrame(most_common_words, columns=['word', 'frequency'])
plt.figure(figsize=(10, 5))
plt.bar(top_words_df['word'], top_words_df['frequency'])
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 10 Words in The Old Man and the Sea')
plt.show()

print(f"Model Mean Squared Error: {mse:.2f}")
