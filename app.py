import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Next Word Prediction | Nihal Ahemad Khan",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------
# Sidebar (Your Professional Info)
# ---------------------------------
st.sidebar.title("👨‍💻 About Me")
st.sidebar.markdown("""
**Nihal Ahemad Khan**   

**Interests:**
* Machine Learning & AI
* Natural Language Processing
* Deep Learning Systems
 
[🔗 LinkedIn](https://www.linkedin.com/in/nihal-ahemad-khan)  
[💻 GitHub](https://github.com/bytecreater)
""")

st.sidebar.divider()
st.sidebar.info("This project demonstrates LSTM-based Next Word Prediction using NLP techniques.")

# ---------------------------------
# Load Model & Files (Cached)
# ---------------------------------
@st.cache_resource
def load_assets():
    try:
        # Load the pre-trained model
        model = load_model("next_word_model.h5", compile=False)
        
        # Load the tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
            
        # Load max_len
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
            
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        return model, tokenizer, max_len, reverse_word_index
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

# Call the function to load assets
model, tokenizer, max_len, reverse_word_index = load_assets()

# ---------------------------------
# Main App UI
# ---------------------------------
st.title("🧠 Next Word Prediction App")
st.write("Enter a phrase below, and the LSTM model will predict the most likely next word.")

# Input section
input_text = st.text_input("Enter your sentence:", placeholder="e.g., How are you")

if st.button("Predict Next Word", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    elif model is None:
        st.error("Model not loaded. Check your file paths.")
    else:
        # Preprocessing
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        
        if len(token_list) == 0:
            st.error("The model doesn't recognize these words. Try different text.")
        else:
            # Pad sequences
            token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
            
            # Predict
            with st.spinner('Thinking...'):
                prediction = model.predict(token_list, verbose=0)
                predicted_index = np.argmax(prediction, axis=-1)[0]
                predicted_word = reverse_word_index.get(predicted_index, "Unknown")

            if predicted_word == "Unknown":
                st.error("Could not determine the next word.")
            else:
                st.success(f"Result: **{input_text}** ... **{predicted_word}**")
                st.balloons()

# ---------------------------------
# Project Highlights
# ---------------------------------
st.divider()
with st.expander("🚀 Project Technical Details"):
    st.markdown("""
    * **Preprocessing:** Text cleaning & Tokenization via Keras.
    * **Architecture:** Long Short-Term Memory (LSTM) layers to capture sequential dependencies.
    * **Logic:** Uses N-gram sequence generation for training data.
    * **Output:** Softmax activation for multi-class word classification.
    """)