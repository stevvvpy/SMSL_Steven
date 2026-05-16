import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

# Menangani data kosong
def handle_missing_values(df):
  # Drop data kosong jika tidak memiliki ID
  df = df.dropna(subset=['StudentID'])

  '''
   Menangani kolom Age, ParentalSupport dan GPA dengan nilai rata-rata
   Menangani kolom Ethnicity kosong menjadi others
   Menangani kolom ParantalEducation kosong menjadi frekuensi tertinggi
   Menangani kolom StudyTimeWeekly dengan nilai 0 dengan asumsi bahwa tidak pernah belajar sehingga tidak mengisinya
   Menangani kolom Absences dengan nilai 0 dengan asumsi bahwa tidak pernah hadir sehingga tidak diabsen
   Menangani kolom Tutoring dengan klasifikasi 0 yaitu tidak mengikuti tutoring
    '''

  df = df.fillna({
    'Age': df['Age'].mean(),
    'ParentalSupport': df['ParentalSupport'].mean(),
    'GPA': df['GPA'].mean(),
    'Ethnicity': 3,
    'ParentalEducation': df['ParentalEducation'].mode()[0],
    'StudyTimeWeekly': 0,
    'Absences': 0,
    'Tutoring': 0
})
  return df

# Menghapus data duplikat
def drop_duplicate(df):
  try:
    df.drop_duplicates(inplace=True)
  except:
    pass
  return df

# Membuang kolom tidak diperlukan seperti StudentID dan Gender untuk mengatasi bias Gender
def drop_column(df):
  try:
      df.drop(['StudentID', 'Gender'], axis=1, inplace=True)
  except:
    pass
  return df

# Deteksi dan penanganan outlier
def handle_outliers(df):
  # Mengubah Age menjadi rata-rata jika terdeteksi outlier
  df['Age'] = np.where((df['Age'] > df['Age'].mean() + 3 * df['Age'].std())| (df['Age'] <df['Age'].mean() - 3 * df['Age'].std()), df['Age'].mean(), df['Age'])
  # Mengubah ParentalSupport, ParantalEducation, ethnicity, StudyTimeWeekly, Absences, dan Tutoring sesuai rentan
  df['ParentalSupport'] = np.where(df['ParentalSupport'] > 4, 4, df['ParentalSupport'])
  df['ParentalSupport'] = np.where(df['ParentalSupport'] < 0, 0, df['ParentalSupport'])

  df['ParentalEducation'] = np.where(df['ParentalEducation'] > 4, 4, df['ParentalEducation'])
  df['ParentalEducation'] = np.where(df['ParentalEducation'] < 0, 0, df['ParentalEducation'])

  df['Ethnicity'] = np.where(df['Ethnicity'] > 3, 3, df['Ethnicity'])
  df['Ethnicity'] = np.where(df['Ethnicity'] < 0, 0, df['Ethnicity'])

  df['StudyTimeWeekly'] = np.where(df['StudyTimeWeekly'] > 20, 20, df['StudyTimeWeekly'])
  df['StudyTimeWeekly'] = np.where(df['StudyTimeWeekly'] < 0, 0, df['StudyTimeWeekly'])

  df['Absences'] = np.where(df['Absences'] > 30, 30, df['Absences'])
  df['Absences'] = np.where(df['Absences'] < 0, 0, df['Absences'])

  df['Tutoring'] = np.where(df['Tutoring'] > 1, 1, df['Tutoring'])
  df['Tutoring'] = np.where(df['Tutoring'] < 0, 0, df['Tutoring'])
  return df

# Menormalisasi Age, ParentalEducation, StudyTimeWeekly, Absences, dan PaerentalSupport
def normalize_features(df):
  scaler = MinMaxScaler()
  fitur = ['Age', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'ParentalSupport']
  df[fitur] = scaler.fit_transform(df[fitur])
  return df

# Membuat one hot encode untuk ethnicity
def encode_feature(df):
  oh_encoder = OneHotEncoder(sparse_output=False, dtype=int)
  encoded_ethnicity = oh_encoder.fit_transform(df[['Ethnicity']])
  nama_etnis = ['Ethnicity_Caucasian', 'Ethnicity_African_American', 'Ethnicity_Asian', 'Ethnicity_Other']
  df_ethnicity_encoded = pd.DataFrame(encoded_ethnicity, columns=nama_etnis)
  df = pd.concat([df, df_ethnicity_encoded], axis=1)
  df.drop('Ethnicity', axis=1, inplace=True)
  return df 

def preprocess(df = pd.read_csv('Student_performance_data_raw/Student_performance_data_raw.csv')):
  try:
    df = drop_duplicate(df)
  except:
    pass
  
  try:
    df = handle_missing_values(df)
  except:
    pass
  
  try:
    df = drop_column(df)
  except:
    pass
  
  try:
    df = handle_outliers(df)
  except:
    pass
  
  try:
    df = normalize_features(df)
  except:
    pass
  
  try:
    df = encode_feature(df)
  except:
    pass
  

  df.to_csv('preprocessing/Student_performance_dataset_preprocessing/cleaned_data.csv', index=False)
  print("Preprocessing selesai!")

if __name__ == '__main__':
  preprocess()