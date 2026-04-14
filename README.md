# AI-Based Speech to Indian Sign Language Translator

This is a major project developed to convert spoken language into
Indian Sign Language using Artificial Intelligence techniques.

## Features
- Speech to text conversion
- NLP-based text preprocessing
- CNN-based hand gesture recognition
- Real-time gesture prediction using webcam
- Streamlit-based user interface

## Technologies Used
- Python
- TensorFlow
- OpenCV
- MediaPipe
- NLP (NLTK)
- Streamlit

## Project Structure
backend/ - core logic  
frontend/ - UI  
dataset/ - training & testing images  
models/ - trained CNN model  
evaluation/ - performance metrics  

## How to Run
1. Activate virtual environment
2. Install dependencies  
   pip install -r requirements.txt
3. Train model  
   python backend/sign_generation/cnn_gesture_model.py
4. Run application  
   streamlit run frontend/app.py