import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from langchain import HuggingFaceHub
import os
from langchain.prompts.prompt import PromptTemplate
import ast
import streamlit as st



os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000000})

df = pd.read_csv(r'C:\Users\Santhosh Sivan\Desktop\earth\source\New earthquake data.csv')
new_df = df.drop(['Year', 'Month', 'Day', 'Time','Timestamp'],axis=1)


label_encoder = LabelEncoder()
new_df['encoded_region'] = label_encoder.fit_transform(new_df['Region'])
new_df = new_df.drop(['Region'],axis=1)

X = new_df.drop(['Mag','encoded_region'],axis=1)
y = new_df['Mag']

# test and train split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize a random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the regressor to the training data
rf.fit(X_train, y_train)

# input_question='What magnitude is expected for an earthquake occurring at latitude 30.7849, longitude 94.6516, and a depth of 10 km?'
order = ['Latitude','Longitude','Depth']
def eq():
    st.write("")
    st.markdown("<p style='text-align: center; color: black; font-size:20px; margin-top: -30px ;'>Earthquake magnitude regression using random forest regressor - Dhamodharan</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;margin-top:0px;background-color:gray;>",unsafe_allow_html=True)
    
    w1,col1,col2,w2=st.columns((1,2,3,0.7))
    w12,col11,col22,w22=st.columns((1,2,3,0.7))
    with col1:
        st.markdown("### ")
        st.write('# ')
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Natural language input</span></p>", unsafe_allow_html=True)
    with col2:
        st.write("")
        st.write("")
        input_question = st.text_input("")
    if input_question!="":
        prompt_template = '''
                            Given a QUESTION you should return the output correctly in the order {order} in the list. Restrict your output to no. of tokens required for arranging the output for the specified QUESTION.
                            Do not strictly write the input question in the output.

                            Examples:
                            question: `Predict the magnitude of an earthquake at latitude 25.9591, longitude 90.6152, and depth 56 km.`; output: [25.9591,90.6152,56]
                            question: `Estimate the earthquake magnitude when the location is at latitude 25.9591, longitude 90.6152, and depth 56 km.`; output: [25.9591,90.6152,56]
                            question: `What is the expected earthquake magnitude at coordinates (25.9591, 90.6152) with a depth of 56 km?`; output: [25.9591,90.6152,56]
                            question: `Calculate the earthquake magnitude for a seismic event at (25.9591, 90.6152) with a depth of 56 km.`; output: [25.9591,90.6152,56]
                            question: `Determine the magnitude of an earthquake occurring at a depth of 56 km with coordinates (25.9591, 90.6152).`; output: [25.9591,90.6152,56]
                            question: `Forecast the earthquake magnitude for a location with latitude 25.9591, longitude 90.6152, and a depth of 56 km.`; output: [25.9591,90.6152,56]
                            question: `Find the expected magnitude of an earthquake at (25.9591, 90.6152) and a depth of 56 km.`; output: [25.9591,90.6152,56]
                            question: `What magnitude can be anticipated for an earthquake at coordinates 25.9591, 90.6152, and depth 56 km?​`; output: [25.9591,90.6152,56]
                            question: `Provide a magnitude prediction for an earthquake with latitude 25.9591, longitude 90.6152, and depth 56 km.`; output: [25.9591,90.6152,56]
                            question: `Estimate the earthquake's magnitude at a depth of 56 km and coordinates (25.9591, 90.6152).`; output: [25.9591,90.6152,56]
                            question: `Given a seismic event at latitude 25.9591, longitude 90.6152, and depth 56 km, what will be the magnitude of the earthquake?`; output: [25.9591,90.6152,56]
                            question: `Predict the earthquake magnitude for a location at (25.9591, 90.6152) with a depth of 56 km.`; output: [25.9591,90.6152,56]
                            question: {input}; output:
                            '''

        prompt = PromptTemplate(input_variables=['order', 'input'], template=prompt_template)
        PROMPT = prompt.format(order = order, input = input_question).strip()

        response = llm.generate([PROMPT]).generations[0][0].text
        string_representation = response
        out_list = ast.literal_eval(string_representation)
        rf.predict([out_list])

        original_list = out_list
        start_value = original_list[2] - 1
        end_value = original_list[2] + 1

        resulting_lists = [[original_list[0], original_list[1], i] for i in range(start_value, end_value + 1)]


        for i in resulting_lists:
            data = pd.DataFrame({'Lat': [i[0]], 'Lon': [i[1]], 'Depth': [i[2]]})
            prediction = rf.predict(data)

        pred = []
        # Assuming feature names are 'Latitude', 'Longitude', 'Depth' in the same order as your data
        for i in resulting_lists:
            data = pd.DataFrame({'Lat': [i[0]], 'Lon': [i[1]], 'Depth': [i[2]]})
            prediction = rf.predict(data)[0]
            rounded_prediction = round(prediction, 3)
            pred.append(prediction)


        formatted_predictions = ', '.join(map(str, pred))

        with col22:
            if st.button('Predict'):
                with col2:
                    st.success(f"The possible outcomes are {formatted_predictions}")
