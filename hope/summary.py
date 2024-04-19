import requests
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Fetch the text from the provided URL
url = "https://gutenberg.ca/ebooks/hemingwaye-oldmanandthesea/hemingwaye-oldmanandthesea-00-t.txt"
response = requests.get(url)
text = response.text

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Process the text using spaCy
doc = nlp(text)

# Tokenize the text into sentences
sentences = [sentence for sentence in doc.sents]

# Calculate word frequency for each word in the document
word_frequencies = {}
for word in doc:
    if word.text.lower() not in STOP_WORDS:
        if word.text.lower() not in word_frequencies:
            word_frequencies[word.text.lower()] = 1
        else:
            word_frequencies[word.text.lower()] += 1

# Calculate the normalized word frequencies
maximum_frequency = max(word_frequencies.values())
for word in word_frequencies:
    word_frequencies[word] = word_frequencies[word] / maximum_frequency

# Calculate sentence scores based on word frequencies
sentence_scores = {}
for sentence in sentences:
    for word in sentence:
        if word.text.lower() in word_frequencies:
            if sentence not in sentence_scores:
                sentence_scores[sentence] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sentence] += word_frequencies[word.text.lower()]

# Get the top sentences for the summary
SENTENCES_COUNT = 3
summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:SENTENCES_COUNT]

# Print the summary
print("Summary:")
for sentence in summary_sentences:
    print(sentence.text)