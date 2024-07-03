import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

# Load the model
file = open("pickle/model.pkl", "rb")
gbc = pickle.load(file)
file.close()

# Define the Streamlit app
def main():
    st.title("Phishing Website Detector")

    url = st.text_input("Enter the URL:")
    if st.button("Check"):
        if url:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)

            y_pred = gbc.predict(x)[0]
            # 1 is safe, -1 is unsafe
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

            # Display the prediction
            if y_pred == 1:
                st.success(f"It is {y_pro_non_phishing*100:.2f}% safe to go")
            else:
                st.warning(f"It is {y_pro_phishing*100:.2f}% unsafe to go")
            st.write(f"Probability of being a non-phishing site: {y_pro_non_phishing:.2f}")
        else:
            st.error("Please enter a URL.")

if __name__ == "__main__":
    main()
