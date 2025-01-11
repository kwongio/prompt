import streamlit as st
from inference import inference_langchain

review = st.text_input("리뷰")

reviews = st.multiselect("리뷰", review.split(",")[:5])
if st.button("submit"):
 summary = inference_langchain(reviews)
 st.write(summary)


# ssl._create_default_https_context = ssl._create_unverified_conte
# url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings"
# df = pd.read_csv(url, sep='\t')
# reviews = st.multiselect("리뷰", df.iloc[:10]['document'])
# if st.button("submit"):
#  summary = inference_langchain(reviews)
#  st.write(summary)
