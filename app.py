from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('knn_for_agent_score')
def predict(model, data_unseen):
    if (data_unseen.iloc[0]["Education Level"] == 'CA'):
        prediction ='Hire'
    else:    
        prediction = predict_model(model, data=data_unseen)
        prediction = prediction.Label[0]
        if (int(prediction)== 1): 
            prediction ='Hire'
        else: 
            prediction ="Don't Hire"
    #predictions_df = predict_model(estimator=model, data=input_df)
    #predictions = predictions_df['Label'][0]
    return prediction

def run():
        st.title("AGENT PRE-SCREENING QUESTIONNAIRE")
        Gender = st.selectbox('Gender', ['Male', 'Female'])
        
        Age = st.selectbox('Age', ['<30','31-40','41-50','>50'])
        
        insurancestartedBracket = st.selectbox("Agent's age when he/she started working in insurance industry", ['18-20','21-25','26-30','31-35','>35'])
        
        Education = st.selectbox('Education', ['UG', 'Grad', 'PG','CA'])
        
        Experience = st.selectbox('Industry Experience', ['<10', '>10'])
        
        Family = st.selectbox('Other Family Member Involved', ['Yes', 'No'])

        output=""

        input_dict = {'Gender' : Gender, 'Age Bracket' : Age, 'Age when insurance started Bracket' : insurancestartedBracket, 'Education Level' : Education, 'Industry Experience bucket' : Experience, 'Other Family Member Involved' : Family}
        #for keys in input_dict.keys():
         #   input_dict[keys] = [str.strip(input_dict[keys])] 
        data_unseen = pd.DataFrame([input_dict])
        #data_unseen.columns = map(str.strip,data_unseen.columns)

        if st.button("Predict"):
            output = predict(model=model, data_unseen=data_unseen)
            output = str(output)

        st.success('Recommendation:  {}'.format(output))

    

if __name__ == '__main__':
    run()