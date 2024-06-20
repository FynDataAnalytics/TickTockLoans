import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('final_model.pkl')

# Function to preprocess data
def preprocess_data(df):
    # Create a dictionary mapping statuses to numerical values
    status_values = {
        'ACTIVE': 1,
        'BANKRUPTCY': 0,
        'CHARGE_OFF': 0,
        'COLLECTION': 0,
        'DELINQUENT': 0,
        'FULL_RECOVERY': 1,
        'PAID_OFF': 1,
        'PARTIAL_RECOVERY': 0,
        'REFINANCED': 0,
        'SOLD': 0
    }

    # Map the status values to the 'tradeline_status' column
    df['status_value'] = df['tradeline_status'].map(status_values)

    # Drop the 'tradeline_status' column
    df = df.drop(columns=['tradeline_status', 'archive'])

    # One-hot encode the 'state' column
    df_encoded = pd.get_dummies(df, columns=['state'], drop_first=True)

    # Identify numerical columns
    numerical_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Remove target and non-feature columns from numerical columns list
    numerical_columns.remove('status_value')
    numerical_columns.remove('id')

    # Standardize numerical columns
    scaler = StandardScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

    return df_encoded

# Load the data
@st.cache_data
def load_data(file):
    # Check the file type
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format")
    
    # Create a dictionary mapping statuses to numerical values
    status_values = {
        'ACTIVE': 1,
        'BANKRUPTCY': 0,
        'CHARGE_OFF': 0,
        'COLLECTION': 0,
        'DELINQUENT': 0,
        'FULL_RECOVERY': 1,
        'PAID_OFF': 1,
        'PARTIAL_RECOVERY': 0,
        'REFINANCED': 0,
        'SOLD': 0
    }

    # Map the status values to the 'tradeline_status' column
    df['status_value'] = df['tradeline_status'].map(status_values)

    # Drop the 'tradeline_status' column
    df = df.drop(columns=['tradeline_status', 'archive'])

    # One-hot encode the 'state' column
    df_encoded = pd.get_dummies(df, columns=['state'], drop_first=True)

    # Identify numerical columns
    numerical_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Remove target and non-feature columns from numerical columns list
    numerical_columns.remove('status_value')
    numerical_columns.remove('id')

    # Standardize numerical columns
    scaler = StandardScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
    
    return df_encoded

# Function to check prediction by id
def check_prediction_by_id(df_encoded, model, id_to_check):
    # Locate the specific id
    row = df_encoded[df_encoded['id'] == id_to_check]
    if row.empty:
        st.error(f"No record found for id: {id_to_check}")
        return None
    
    # Drop 'status_value' and 'id' columns for prediction
    row_features = row.drop(['status_value', 'id'], axis=1)
    
    # Predict probabilities
    probabilities = model.predict_proba(row_features)[0]
    probability_yes = probabilities[1] * 100
    probability_no = probabilities[0] * 100
    
    if probability_yes > probability_no:
        result = f"Probability that id {id_to_check} falls into 'Yes': {probability_yes:.2f}%"
    else:
        result = f"Probability that id {id_to_check} falls into 'No': {probability_no:.2f}%"
    
    return result

# Main Streamlit app
def main():
    st.title("Prediction App")
    st.write("Upload your data and enter an ID to get the probability of falling into 'Yes' or 'No' category.")
    
    # Load the model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Load data
            df_encoded = load_data(uploaded_file)
            
            # Show head of uploaded data
            st.subheader("Head of Uploaded Data")
            st.write(df_encoded.head())
            
            # Input field for id
            id_to_check = st.text_input("Enter ID:", "")
            
            # Check prediction when button is clicked
            if st.button("Check Prediction"):
                if id_to_check.isdigit():
                    id_to_check = int(id_to_check)
                    result = check_prediction_by_id(df_encoded, model, id_to_check)
                    if result:
                        st.success(result)
                else:
                    st.error("Please enter a valid numeric ID.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
