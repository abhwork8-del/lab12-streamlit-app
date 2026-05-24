import streamlit as st
st.set_page_config(page_title="Lab 12 App", layout="centered")
st.title("🚀 Lab 12 Streamlit Deployment")
st.write("App is successfully running from GitHub!")
name = st.text_input("Enter your name")
if name:
    st.success(f"Hello {name} 👋 Welcome to Lab 12")
