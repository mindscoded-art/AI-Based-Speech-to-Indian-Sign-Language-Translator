from speech.speech_to_text import speech_to_text
from nlp.text_processing import preprocess_text

print("Speak something...")

spoken_text = speech_to_text()
processed_text = preprocess_text(spoken_text)

print("Original Text:", spoken_text)
print("Processed Text:", processed_text)