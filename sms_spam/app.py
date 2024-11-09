import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import pickle
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input=st.text_area('enter text')







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


if st.button('predict'):
    data=text_convert(input)
    vec=tfidf.transform([data])
    result=model.predict(vec)[0]
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")

if st.button('RESET'):
    st.empty()