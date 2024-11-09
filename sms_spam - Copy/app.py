import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classsifier")


def text_convert(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)


    text =y[:]
    y.clear()

    for i in text:
        
        y.append(ps.stem(i))
        
    return " ".join(y)
















sms=st.text_area("Enter your message")




if st.button('submit'):
        text=text_convert(sms)
        vec=tfidf.transform([text])

        result=model.predict(vec)[0]

        if result ==1:
            st.header("message is Spam")
        else:
            st.header("Not spam")




   

