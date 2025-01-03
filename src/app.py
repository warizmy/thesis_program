import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
MODEL_PATH = './models/knn.joblib' 
SCALER_PATH = './models/scaler.pkl' 
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(
    page_title="Akselerasi Skripsi Mahasiswa Universitas Bumigora", 
    page_icon="./img/assets/favicon.ico" 
)


st.title("Sistem Seleksi Program Akselerasi Skripsi")
st.write("Aplikasi ini digunakan untuk menentukan mahasiswa yang memenuhi kriteria program akselerasi berdasarkan data yang diunggah.")

uploaded_file = st.file_uploader("Unggah file CSV Anda", type="csv")

if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data yang Diupload:")
        st.dataframe(data.head(50), width=800)

        if not all(col in data.columns for col in ['nim', 'ipk', 'total_sks']):
            raise ValueError("File CSV harus memiliki kolom 'nim', 'ipk', dan 'total_sks'.")

        # Preprocessing: Extract relevant features
        features = data[['ipk', 'total_sks']]
        scaled_features = scaler.transform(features)  # Normalize features
        
        # Prediction
        predictions = model.predict(scaled_features)
        
        data['Prediksi'] = predictions
        
        # Format NIM (remove commas if exist)
        data['nim'] = data['nim'].astype(str).str.replace(',', '')
        
        # Display predictions in a table
        st.write("Hasil Prediksi:")
        st.dataframe(data[['nim', 'Prediksi']], width=800)  # Show the dataframe with predictions
        
        # Filter students eligible for the program
        eligible_students = data[data['Prediksi'] == 'Ya'][['nim']]
        
        # Display eligible students in a table
        st.write("Mahasiswa yang Memenuhi Kriteria Program Akselerasi:")
        st.dataframe(eligible_students, width=800)
        
        total_eligible = len(eligible_students)
        st.write(f"Jumlah Mahasiswa yang Memenuhi Kriteria Program Akselerasi: {total_eligible}")
        
    except ValueError as ve:
        st.error(f"Error: {ve}")
    except Exception as e:
        st.error("Terjadi kesalahan saat memproses file. Pastikan format file sesuai.")
