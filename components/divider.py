import streamlit as st

def divider(sidebar=False):
    if sidebar:
        return st.sidebar.markdown("<hr style='margin-top:1px;margin-bottom:1px;'>",unsafe_allow_html=True)
    else:
        return st.markdown("<hr style='margin-top:1px;margin-bottom:1px;'>",unsafe_allow_html=True)