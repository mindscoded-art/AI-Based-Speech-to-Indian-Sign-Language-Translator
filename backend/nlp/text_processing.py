import nltk
nltk.download('punkt')

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    stop_words = ["is", "am", "are", "the", "to"]
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)