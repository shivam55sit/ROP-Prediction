import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
from io import BytesIO

# Load the pre-trained pipeline
pipe = pickle.load(open('pipe.joblib', 'rb'))

# Load the dataframe (for select box choices)
df = pickle.load(open('final_dataframe_rop.sav', 'rb'))

# Display the logo
st.logo('lvpei.jfif', size="large")  

st.title("ROP Prediction")

#columns in original dataframe for predictio -
#['NICU_rating', 'Gender', 'Birth_Weight', 'O2_Therapy', 'Resp_Distress',
      # 'Septicemia', 'Blood_Transfusion', 'GA_Days', 'Age', 'Delivery_Type',
       #'Preterm_severity']

# File upload section for batch processing
uploaded_file = st.file_uploader("Upload an Excel file for batch prediction", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file into a DataFrame
    input_df = pd.read_excel(uploaded_file)

    # Ensure the necessary columns are present in the uploaded file
    required_columns = ['Birth_Weight', 'Resp_Distress', 'Septicemia', 'Blood_Transfusion', 
                        'GA_Days', 'NICU_rating', 'Gender', 'O2_Therapy', 'Age', 'Delivery_Type', 'Preterm_severity']
    
    if all(col in input_df.columns for col in required_columns):
        # Preprocess the inputs from the file
        input_df['Resp_Distress'] = input_df['Resp_Distress'].apply(lambda x: 1 if x == 'Y' else 0)
        input_df['Septicemia'] = input_df['Septicemia'].apply(lambda x: 1 if x == 'Y' else 0)
        input_df['Blood_Transfusion'] = input_df['Blood_Transfusion'].apply(lambda x: 1 if x == 'Y' else 0)
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
        input_df['O2_Therapy'] = input_df['O2_Therapy'].apply(lambda x: 1 if x == 'Y' else 0)
        input_df['Delivery_Type'] = input_df['Delivery_Type'].apply(lambda x: 1 if x == 'Pre Term' else 0)


        # Handle NaN values in NICU_rating by filling them with 1
        input_df['NICU_rating'].fillna(1, inplace=True)
        input_df['Delivery_Type'].fillna(0, inplace=True) # replacing null value in delivery type with 0 i.e. full term
        
        # Assign preterm severity based on GA_Days
        input_df['Preterm_severity'] = 0
        input_df.loc[input_df['GA_Days'] < 196, 'Preterm_severity'] = 3
        input_df.loc[(input_df['GA_Days'] >= 196) & (input_df['GA_Days'] < 224), 'Preterm_severity'] = 2
        input_df.loc[(input_df['GA_Days'] >= 224) & (input_df['GA_Days'] <= 259), 'Preterm_severity'] = 1
      
        # Make predictions for the entire batch
        y_prob_batch = pipe.predict_proba(input_df)

        # Get predicted classes and confidence scores
        predicted_classes = np.argmax(y_prob_batch, axis=1)
        confidence_scores = np.max(y_prob_batch, axis=1)

        # Add predictions and confidence to the DataFrame
        input_df['Predicted_ROP'] = np.where(predicted_classes == 1, 'ROP', 'No ROP')
        input_df['Confidence'] = confidence_scores * 100

        # Display the results as a table
        st.write("Batch Predictions")
        st.dataframe(input_df[['Predicted_ROP', 'Confidence']])

        # Option to download the results as an Excel file
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            processed_data = output.getvalue()
            return processed_data

        result_excel = convert_df_to_excel(input_df)

        st.download_button(label="Download Predictions",
                           data=result_excel,
                           file_name="batch_predictions_ROP.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error(f"The uploaded file does not contain the required columns: {', '.join(required_columns)}")

else:
    st.write("Or manually input values for a single prediction.")

    # Input fields for manual prediction (single instance)
    birth_weight = st.number_input('Birth Weight (Grams)', value=1000)
    Resp_Distress = st.selectbox('Respiratory Distress', ['No', 'Yes'])
    Septicemia = st.selectbox('Septicemia', ['No', 'Yes'])
    Blood_Transfusion = st.selectbox('Blood Transfusion', ['No', 'Yes'])
    O2_Therapy = st.selectbox('O2 Therapy', ['No', 'Yes'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    GA_Days = st.number_input('Gestation Period (GA) in Days', value=240)
    PMA_Days = st.number_input('Postmenstrual Age (PMA) in Days', value=260)
    nicu = st.selectbox('NICU Rating', df['NICU_rating'].unique())
    Delivery_Type = st.selectbox('Delivery Type', df['Delivery_Type'].unique())
    Age = np.abs(PMA_Days-GA_Days)

    # Preprocessing the input when 'Predict' button is pressed
    if st.button('Predict'):
        # Convert categorical inputs to numeric
        Delivery_Type = 1 if Delivery_Type == 'Pre Term' else 0
        Resp_Distress = 1 if Resp_Distress == 'Yes' else 0
        Septicemia = 1 if Septicemia == 'Yes' else 0
        Blood_Transfusion = 1 if Blood_Transfusion == 'Yes' else 0
        Gender = 1 if Gender == 'Male' else 0
        O2_Therapy = 1 if O2_Therapy == 'Yes' else 0
        if GA_Days < 196:
            Preterm_severity = 3
        elif 196 <= GA_Days < 224:
            Preterm_severity = 2
        elif 224 <= GA_Days <= 259:
            Preterm_severity = 1
        else:
            Preterm_severity = 0
        
        # Handle NaN values for NICU
        nicu = 1 if pd.isna(nicu) else nicu 

        # Create a DataFrame from the input values
        input_data = pd.DataFrame({
            'NICU_rating': [nicu],
            'Birth_Weight': [birth_weight],
            'Resp_Distress': [Resp_Distress],
            'Septicemia': [Septicemia],
            'Blood_Transfusion': [Blood_Transfusion],
            'O2_Therapy': [O2_Therapy],
            'GA_Days': [GA_Days],
            'Age': [Age],
            'PMA_Days': [PMA_Days],
            'Gender': [Gender],
            'Preterm_severity': [Preterm_severity],
            'Delivery_Type' : [Delivery_Type]
        })

        # Make the prediction using predict_proba
        y_prob = pipe.predict_proba(input_data)

        # Predicted class and confidence score
        predicted_class = np.argmax(y_prob, axis=1)[0]
        confidence_score = np.max(y_prob, axis=1)[0]

        # Display the result with confidence
        if predicted_class == 0:
            st.title(f"NO ROP.")
            st.write(f"Confidence: {confidence_score*100:.2f}%")
            st.image("thumbs-up.webp")
        else:
            st.title(f"ROP symptoms are significant. Please consult Dr. Tapas!")
            st.write(f"Confidence: {confidence_score*100:.2f}%")
            st.image("thumb-down.jpg")
