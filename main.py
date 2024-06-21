import streamlit as st
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import base64
from streamlit_option_menu import option_menu

# Function to set the background image
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background
set_background('asset/bg.png')


# Load SVM model
try:
    with open('model/svc_model.pkl', 'rb') as file:
        svc_model = pickle.load(file)
    # st.write("SVM model loaded successfully.")
except Exception as e:
    print(f"Error loading SVM model: {e}")

# Load RNN model

# try:
    
    # st.write("RNN model loaded successfully.")
# except Exception as e:
#     print(f"Error loading RNN model: {e}")

# Load tokenizer
try:
    with open('model/tokenizer.json', 'r') as file:
        tokenizer_json = json.load(file)
        tokenizer = tokenizer_from_json(tokenizer_json)
    # st.write("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Load vectorizer
try:
    with open('model/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    # st.write("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

# Function to predict sentiment using SVC
def predict_sentiment_svc(text):
    text_vectorized = vectorizer.transform([text])
    prediction = svc_model.predict(text_vectorized)
    return prediction[0]

# Function to preprocess text for RNN
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)  # Use appropriate maxlen
    return padded

# Function to predict sentiment using RNN
def predict_sentiment_rnn(text):
    rnn_model = load_model('model/rnn_model.keras')
    preprocessed_text = preprocess_text(text)
    prediction = rnn_model.predict(preprocessed_text)
    return np.argmax(prediction, axis=1)[0]

# Streamlit app
st.title('Mental Health Sentiment Analysis')

# Sidebar menu
with st.sidebar:
    selected = option_menu("Menu", ["Overview", "Predict"],
                           icons=['house', 'activity'], menu_icon="cast", default_index=0)

if selected == "Overview":
    exec(open("overview.py").read())

elif selected == "Predict":
    st.header("Predict Sentiment")

    # Function to compare model accuracies
    def compare_models():
        # Dummy accuracy values (replace with actual values)
        st.header("Accuracy Comparison")
        rnn_accuracy = 0.86  # Example value
        svc_accuracy = 0.80  # Example value

        st.write(f"RNN model accuracy: **{rnn_accuracy}**")
        st.write(f"SVM model accuracy: **{svc_accuracy}**")
        st.title("Comparison between RNN and Linear SVC")

        st.header("RNN (Recurrent Neural Network)")
        st.markdown("""
        1. **Internal Memory**: RNNs have an internal memory structure that allows them to remember information from previous steps in the data sequence. This makes RNNs very effective at capturing long-term dependencies and temporal patterns.
        2. **Contextual Understanding**: RNNs can leverage the context from previous data to make more accurate predictions in sequential tasks such as text analysis or time series.
        3. **Sequence Structure**: RNNs are specifically designed to work with data that has a sequential structure, allowing them to understand and process data in the given order.
        """)

        st.header("Linear SVC (Support Vector Classifier)")
        st.markdown("""
        1. **Linear Model**: Linear SVC is a classification model that separates data based on a linear hyperplane. It treats each feature in the data as independent and does not consider the order or context.
        2. **Context Limitation**: Linear SVC lacks a mechanism to capture temporal relationships or sequential context in the data, making it less effective for tasks requiring sequence understanding like text analysis or time series.
        3. **Simple and Fast**: Linear SVC is simpler and faster to train compared to RNNs but is less capable of handling data with sequential or temporal dependencies.
        """)

    # Questions to ask
    questions = [
        "What feelings have been dominating you lately?",
        "How has your sleep been over the past few weeks?",
        "What are you most worried about right now?",
        "How do you feel about your work or school?",
        "Do you feel supported by those around you?"
    ]

    responses = []

    # Loop through questions
    for question in questions:
        st.write(question)
        user_input = st.text_input("Enter your answer here (please use English):", key=question)
        if st.button("Check Sentiment", key=f"button_{question}"):
            sentiment_svc = predict_sentiment_svc(user_input)
            sentiment_rnn = predict_sentiment_rnn(user_input)

            sentiment_mapping = {0: "Very Normal", 1: "Normal", 2: "Less Normal", 3: "Bad"}
            svc_sentiment_text = sentiment_mapping.get(sentiment_svc, "Unknown")
            rnn_sentiment_text = sentiment_mapping.get(sentiment_rnn, "Unknown")

            st.write(f"SVC Sentiment: {svc_sentiment_text}")
            st.write(f"RNN Sentiment: {rnn_sentiment_text}")
            responses.append((question, user_input, svc_sentiment_text, rnn_sentiment_text))

    if st.button("Compare Model Accuracies"):
        compare_models()

# import streamlit as st
# import pickle
# import json
# import time
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# import numpy as np
# import base64
# from streamlit_option_menu import option_menu

# # Function to set the background image
# def set_background(png_file):
#     try:
#         with open(png_file, "rb") as f:
#             data = f.read()
#         bin_str = base64.b64encode(data).decode()
#         page_bg_img = f'''
#         <style>
#         .stApp {{
#         background-image: url("data:image/png;base64,{bin_str}");
#         background-size: cover;
#         }}
#         </style>
#         '''
#         st.markdown(page_bg_img, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error setting background image: {e}")

# # Set the background
# set_background('asset/bg.png')

# # Load SVM model
# try:
#     with open('model/svc_model.pkl', 'rb') as file:
#         svc_model = pickle.load(file)
#     st.write("SVM model loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading SVM model: {e}")

# # Load RNN model
# rnn_model = None
# try:
#     rnn_model = load_model('model/rnn_model.keras')
#     st.write("RNN model loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading RNN model: {e}")
#     rnn_model = None

# # Load tokenizer
# try:
#     with open('model/tokenizer.json', 'r') as file:
#         tokenizer_json = json.load(file)
#         tokenizer = tokenizer_from_json(tokenizer_json)
#     st.write("Tokenizer loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading tokenizer: {e}")

# # Load vectorizer
# try:
#     with open('model/vectorizer.pkl', 'rb') as file:
#         vectorizer = pickle.load(file)
#     st.write("Vectorizer loaded successfully.")
# except Exception as e:
#     st.error(f"Error loading vectorizer: {e}")

# # Function to predict sentiment using SVC
# def predict_sentiment_svc(text):
#     try:
#         text_vectorized = vectorizer.transform([text])
#         prediction = svc_model.predict(text_vectorized)
#         return prediction[0]
#     except Exception as e:
#         st.error(f"Error in SVC prediction: {e}")
#         return None

# # Function to preprocess text for RNN
# def preprocess_text(text):
#     try:
#         sequences = tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(sequences, maxlen=100)  # Use appropriate maxlen
#         return padded
#     except Exception as e:
#         st.error(f"Error in text preprocessing: {e}")
#         return None

# # Function to predict sentiment using RNN
# def predict_sentiment_rnn(text):
#     if rnn_model is None:
#         st.error("RNN model is not loaded.")
#         return None
    
#     try:
#         preprocessed_text = preprocess_text(text)
#         if preprocessed_text is not None:
#             time.sleep(0.1)  # Add a small delay
#             prediction = rnn_model.predict(preprocessed_text)
#             return np.argmax(prediction, axis=1)[0]
#         else:
#             return None
#     except BrokenPipeError as e:
#         st.error(f"Broken pipe error in RNN prediction: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Error in RNN prediction: {e}")
#         return None

# # Streamlit app
# st.title('Mental Health Sentiment Analysis')

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu("Menu", ["Overview", "Predict"],
#                            icons=['house', 'activity'], menu_icon="cast", default_index=0)

# if selected == "Overview":
#     exec(open("overview.py").read())

# elif selected == "Predict":
#     st.header("Predict Sentiment")

#     # Function to compare model accuracies
#     def compare_models():
#         # Dummy accuracy values (replace with actual values)
#         st.header("Accuracy Comparison")
#         rnn_accuracy = 0.86  # Example value
#         svc_accuracy = 0.80  # Example value

#         st.write(f"RNN model accuracy: **{rnn_accuracy}**")
#         st.write(f"SVM model accuracy: **{svc_accuracy}**")
#         st.title("Comparison between RNN and Linear SVC")

#         st.header("RNN (Recurrent Neural Network)")
#         st.markdown("""
#         1. **Internal Memory**: RNNs have an internal memory structure that allows them to remember information from previous steps in the data sequence. This makes RNNs very effective at capturing long-term dependencies and temporal patterns.
#         2. **Contextual Understanding**: RNNs can leverage the context from previous data to make more accurate predictions in sequential tasks such as text analysis or time series.
#         3. **Sequence Structure**: RNNs are specifically designed to work with data that has a sequential structure, allowing them to understand and process data in the given order.
#         """)

#         st.header("Linear SVC (Support Vector Classifier)")
#         st.markdown("""
#         1. **Linear Model**: Linear SVC is a classification model that separates data based on a linear hyperplane. It treats each feature in the data as independent and does not consider the order or context.
#         2. **Context Limitation**: Linear SVC lacks a mechanism to capture temporal relationships or sequential context in the data, making it less effective for tasks requiring sequence understanding like text analysis or time series.
#         3. **Simple and Fast**: Linear SVC is simpler and faster to train compared to RNNs but is less capable of handling data with sequential or temporal dependencies.
#         """)

#     # Questions to ask
#     questions = [
#         "What feelings have been dominating you lately?",
#         "How has your sleep been over the past few weeks?",
#         "What are you most worried about right now?",
#         "How do you feel about your work or school?",
#         "Do you feel supported by those around you?"
#     ]

#     responses = []

#     # Loop through questions
#     for question in questions:
#         st.write(question)
#         user_input = st.text_input("Enter your answer here (please use English):", key=question)
#         if st.button("Check Sentiment", key=f"button_{question}"):
#             if user_input:
#                 sentiment_svc = predict_sentiment_svc(user_input)
#                 sentiment_rnn = predict_sentiment_rnn(user_input)

#                 sentiment_mapping = {0: "Very Normal", 1: "Normal", 2: "Less Normal", 3: "Bad"}
#                 svc_sentiment_text = sentiment_mapping.get(sentiment_svc, "Unknown")
#                 rnn_sentiment_text = sentiment_mapping.get(sentiment_rnn, "Unknown")

#                 st.write(f"SVC Sentiment: {svc_sentiment_text}")
#                 st.write(f"RNN Sentiment: {rnn_sentiment_text}")
#                 responses.append((question, user_input, svc_sentiment_text, rnn_sentiment_text))

#     if st.button("Compare Model Accuracies"):
#         compare_models()
