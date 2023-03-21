import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv(r"C:\Users\ADMIN\Desktop\Data_Science_Project\Project_4\emails.csv.xls", encoding= 'latin-1')
data = data[["content", "Class"]]
x = np.array(data["content"])
y = np.array(data["Class"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
import streamlit as st
st.title("Abusive Mail Detection System")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(https://www.spcdn.org/images/Spam-filter.jpg);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url() 

def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
spamdetection()
#adding a button

#if st.button('Submit'):

 # st.write() #displayed when the button is clicked
#else:
 # st.write()