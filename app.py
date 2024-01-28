import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")
import math
from PIL import Image
import source.classification as classification

with open('style/final.css') as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
imcol1, imcol2, imcol3 = st.columns((2,5,3))
with imcol1:
    st.write("")
with imcol2:
    st.image('image/vit.png')
    st.markdown("")
with imcol3:
    st.write("")
#---------Side bar-------#

with st.sidebar:
    selected = st.selectbox("",
                     ['Select Application',"Earth Quake Magnitude Regression"],key='text')
    Library = st.selectbox("",
                     ["Library Used","Streamlit","Image"],key='text1')
    # Gcp_cloud = st.selectbox("",
    #                  ["GCP Services Used","VM Instance","Computer Engine","Cloud Storage"],key='text2')
    # GPT_TOOL =  st.selectbox(" ",('Models Used','GPT3 - Davinci','GPT-3.5 Turbo Model'),key='text3')
    st.markdown("## ")
    href = """<form action="#">
            <input type="submit" value="Clear/Reset" />
            </form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)
    
#--------------function calling-----------#
if __name__ == "__main__":
    # try:
        if selected == "Select Application":
            pass
            st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
        if selected == "Earth Quake Magnitude Regression":
            classification.eq()
    # except BaseException as error:
    #     st.error(error)