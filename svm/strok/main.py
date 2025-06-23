# Import library pandas untuk manipulasi dan analisis data
import pandas as pd
# Import library numpy untuk komputasi numerik
import numpy as np
# Import matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Import seaborn untuk visualisasi statistik
import seaborn as sns
# Import sklearn untuk machine learning
import sklearn
# Import StandardScaler untuk normalisasi data dan LabelEncoder untuk encoding kategorikal
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Import metrik evaluasi model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# Import train_test_split untuk membagi data
from sklearn.model_selection import train_test_split
# Import warnings untuk menangani peringatan
import warnings
# Mengabaikan peringatan
warnings.filterwarnings('ignore')
# Import SVM dari sklearn
from sklearn import svm
# Import ListedColormap untuk visualisasi
from matplotlib.colors import ListedColormap

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data():
    print("=== Memuat Data ===")
    # Membaca file CSV stroke
    df = pd.read_csv('dataset/stroke.csv')
    
    # Menghapus kolom id karena tidak diperlukan untuk analisis
    df = df.drop('id', axis=1)
    
    # Mengisi nilai BMI yang hilang dengan nilai rata-rata
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    
    # Mengubah variabel kategorikal menjadi numerik menggunakan LabelEncoder
    le = LabelEncoder()
    # Daftar kolom kategorikal yang akan di-encode
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    # Melakukan encoding untuk setiap kolom kategorikal
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    
    return df

# Fungsi untuk melatih model SVM
def train_model(df):
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Standardisasi fitur menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Membagi data menjadi data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Membuat dan melatih model SVM dengan kernel RBF
    clf = svm.SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    
    return clf, scaler, X_train, X_test, y_train, y_test

# Fungsi untuk menampilkan Confusion Matrix
def show_confusion_matrix(y_test, y_pred):
    # Membuat figure dengan ukuran 8x6
    plt.figure(figsize=(8, 6))
    # Menghitung confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Menampilkan heatmap confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Fungsi untuk menampilkan Heatmap Korelasi
def show_correlation_heatmap(df):
    # Membuat figure dengan ukuran 10x8
    plt.figure(figsize=(10, 8))
    # Menampilkan heatmap korelasi
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriks Korelasi')
    plt.tight_layout()
    plt.show()

# Fungsi untuk menampilkan Distribusi Fitur
def show_feature_distribution(df):
    # Memilih kolom numerik untuk visualisasi
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']
    # Membuat figure dengan ukuran 12x4
    plt.figure(figsize=(12, 4))
    # Membuat subplot untuk setiap fitur numerik
    for i, kolom in enumerate(numeric_columns, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data=df, x=kolom, hue='stroke', bins=30)
        plt.title(f'Distribusi {kolom}')
    plt.tight_layout()
    plt.show()

# Fungsi untuk menampilkan Visualisasi Prediksi
def show_prediction_visualization(X_train, X_test, y_train, y_test):
    # Membuat figure dengan ukuran 15x5
    plt.figure(figsize=(15, 5))
    
    # Plot data latih
    plt.subplot(1, 2, 1)
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='winter')
    plt.title('Training Set (2 Fitur Pertama)')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    
    # Plot data uji
    plt.subplot(1, 2, 2)
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='winter')
    plt.title('Test Set (2 Fitur Pertama)')
    plt.xlabel(X_test.columns[0])
    plt.ylabel(X_test.columns[1])
    plt.tight_layout()
    plt.show()

# Fungsi untuk menampilkan Decision Boundary
def show_decision_boundary(clf, X_train, X_test, y_train, y_test):
    try:
        # Membuat figure dengan ukuran 15x5
        plt.figure(figsize=(15, 5))
        
        # Plot untuk data latih
        plt.subplot(1, 2, 1)
        X_set, y_set = X_train.iloc[:, :2].values, y_train.values
        
        # Membuat grid untuk visualisasi
        h = 0.02  # ukuran langkah dalam grid
        x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
        y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Menyiapkan titik-titik grid untuk prediksi
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Menambahkan nol untuk fitur lainnya
        mesh_points = np.hstack([mesh_points, np.zeros((mesh_points.shape[0], X_train.shape[1]-2))])
        
        # Melakukan prediksi
        Z = clf.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('red', 'green')))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
        # Plot titik-titik data latih
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                       c=ListedColormap(('red', 'green'))(i), label=j)
        
        plt.title('SVM Decision Boundary (Training set)')
        plt.xlabel(X_train.columns[0])
        plt.ylabel(X_train.columns[1])
        plt.legend()
        
        # Plot untuk data uji
        plt.subplot(1, 2, 2)
        X_set, y_set = X_test.iloc[:, :2].values, y_test.values
        
        # Membuat grid untuk data uji
        x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
        y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Menyiapkan titik-titik grid untuk prediksi
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points = np.hstack([mesh_points, np.zeros((mesh_points.shape[0], X_test.shape[1]-2))])
        
        # Melakukan prediksi
        Z = clf.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('red', 'green')))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
        # Plot titik-titik data uji
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                       c=ListedColormap(('red', 'green'))(i), label=j)
        
        plt.title('SVM Decision Boundary (Test set)')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error dalam visualisasi decision boundary: {str(e)}")
        print("Mencoba metode visualisasi alternatif...")
        
        # Visualisasi alternatif yang lebih sederhana
        plt.figure(figsize=(15, 5))
        
        # Plot data latih
        plt.subplot(1, 2, 1)
        plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='winter')
        plt.title('Training Set (2 Fitur Pertama)')
        plt.xlabel(X_train.columns[0])
        plt.ylabel(X_train.columns[1])
        
        # Plot data uji
        plt.subplot(1, 2, 2)
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='winter')
        plt.title('Test Set (2 Fitur Pertama)')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        
        plt.tight_layout()
        plt.show()

# Fungsi untuk menampilkan Pairplot
def show_pairplot(df):
    # Memilih kolom numerik untuk pairplot
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']
    # Membuat pairplot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df[numeric_columns + ['stroke']], hue='stroke', diag_kind='kde')
    plt.suptitle('Pairplot Fitur Numerik', y=1.02)
    plt.show()

# Fungsi untuk melakukan prediksi
def make_prediction(clf, scaler, gender, age, hypertension, heart_disease, ever_married, 
                   work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    # Mengubah input menjadi array numpy
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married,
                           work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
    # Standardisasi input
    input_scaled = scaler.transform(input_data)
    # Melakukan prediksi
    prediksi = clf.predict(input_scaled)
    return "Berisiko Stroke" if prediksi[0] == 1 else "Tidak Berisiko Stroke"

# Fungsi untuk menampilkan menu
def tampilkan_menu():
    print("\n=== MENU ANALISIS STROKE ===")
    print("1. Tampilkan Confusion Matrix")
    print("2. Tampilkan Heatmap Korelasi")
    print("3. Tampilkan Distribusi Fitur")
    print("4. Tampilkan Visualisasi Prediksi")
    print("5. Tampilkan Decision Boundary")
    print("6. Tampilkan Pairplot")
    print("7. Lakukan Prediksi Baru")
    print("8. Keluar")

# Fungsi utama program
def main():
    # Memuat dan memproses data
    df = load_and_preprocess_data()
    
    # Menampilkan informasi dataset
    print("\n=== INFORMASI DATASET ===")
    print(f"Jumlah baris: {df.shape[0]}")
    print(f"Jumlah kolom: {df.shape[1]}")
    print("\nNama-nama kolom:")
    print(df.columns.tolist())
    print("\n5 Data Pertama:")
    print(df.head())
    
    # Menampilkan statistik deskriptif
    print("\n=== STATISTIK DESKRIPTIF ===")
    print(df.describe())
    
    # Melatih model
    clf, scaler, X_train, X_test, y_train, y_test = train_model(df)
    
    # Menampilkan informasi data latih dan uji
    print("\n=== INFORMASI DATA LATIH DAN UJI ===")
    print(f"Jumlah data latih: {len(X_train)}")
    print(f"Jumlah data uji: {len(X_test)}")
    
    # Menampilkan contoh data latih
    print("\n=== CONTOH DATA LATIH (5 DATA PERTAMA) ===")
    print(pd.DataFrame(X_train, columns=X_train.columns).head())
    
    # Melakukan prediksi awal
    y_pred = clf.predict(X_test)
    
    # Menampilkan metrik evaluasi
    print("\n=== METRIK EVALUASI MODEL ===")
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    
    # Menampilkan contoh hasil prediksi
    print("\n=== CONTOH HASIL PREDIKSI (10 DATA PERTAMA) ===")
    hasil_prediksi = pd.DataFrame({
        'Aktual': y_test[:10],
        'Prediksi': y_pred[:10]
    })
    print(hasil_prediksi)
    
    # Menampilkan nilai y_test dan y_pred
    print("\n=== NILAI Y_TEST ===")
    print(y_test)
    print("\n=== NILAI Y_PRED ===")
    print(y_pred)
    
    # Loop utama program
    while True:
        tampilkan_menu()
        pilihan = input("\nPilih menu (1-8): ")
        
        # Menangani pilihan menu
        if pilihan == '1':
            show_confusion_matrix(y_test, y_pred)
        elif pilihan == '2':
            show_correlation_heatmap(df)
        elif pilihan == '3':
            show_feature_distribution(df)
        elif pilihan == '4':
            show_prediction_visualization(X_train, X_test, y_train, y_test)
        elif pilihan == '5':
            show_decision_boundary(clf, X_train, X_test, y_train, y_test)
        elif pilihan == '6':
            show_pairplot(df)
        elif pilihan == '7':
            # Input data pasien untuk prediksi
            print("\nMasukkan data pasien:")
            gender = int(input("Gender (0:Female, 1:Male): "))
            age = float(input("Age: "))
            hypertension = int(input("Hypertension (0:No, 1:Yes): "))
            heart_disease = int(input("Heart Disease (0:No, 1:Yes): "))
            ever_married = int(input("Ever Married (0:No, 1:Yes): "))
            work_type = int(input("Work Type (0-4): "))
            residence_type = int(input("Residence Type (0:Rural, 1:Urban): "))
            avg_glucose_level = float(input("Average Glucose Level: "))
            bmi = float(input("BMI: "))
            smoking_status = int(input("Smoking Status (0-3): "))
            
            # Melakukan prediksi
            hasil = make_prediction(clf, scaler, gender, age, hypertension, heart_disease,
                                  ever_married, work_type, residence_type, avg_glucose_level,
                                  bmi, smoking_status)
            print("\nPrediksi:", hasil)
        elif pilihan == '8':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        else:
            print("\nPilihan tidak valid. Silakan pilih 1-8.")

# Menjalankan program jika file dijalankan langsung
if __name__ == "__main__":
    main()