import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm

# Memuat data
print("=== Memuat Data ===")
df = pd.read_csv('dataset/diabetes.csv')

# Menampilkan 5 data pertama
print("\n=== 5 Data Pertama ===")
print(df.head())

# Menampilkan statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Memeriksa nilai yang hilang
print("\n=== Pemeriksaan Nilai Hilang ===")
print("Apakah ada nilai yang hilang?", df.isnull().values.any())

# Rekayasa Fitur
print("\n=== Rekayasa Fitur ===")
kolom_tidak_boleh_nol = ["Glucose", "BloodPressure", "SkinThickness"]

for kolom in kolom_tidak_boleh_nol:
    df[kolom] = df[kolom].replace(0, np.nan)
    rata_rata = int(df[kolom].mean(skipna=True))
    df[kolom] = df[kolom].replace(np.nan, rata_rata)

# Membuat visualisasi distribusi fitur
plt.figure(figsize=(12, 8))
for i, kolom in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=kolom, hue='Outcome', bins=30)
    plt.title(f'Distribusi {kolom}')
plt.tight_layout()
plt.savefig('distribusi_fitur.png')
plt.close()

# Membuat heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.tight_layout()
plt.savefig('korelasi.png')
plt.close()

# Memisahkan fitur dan target
X = df.iloc[:, :-1]  # Semua kolom kecuali Outcome
y = df.iloc[:, -1]   # Kolom Outcome

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Membuat dan melatih model SVM
print("\n=== Pelatihan Model SVM ===")
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Melakukan prediksi
y_pred = clf.predict(X_test)

# Evaluasi model
print("\n=== Evaluasi Model ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nMatriks Konfusi:")
print(confusion_matrix(y_test, y_pred))

# Menghitung F1 Score
f1 = f1_score(y_test, y_pred)
print("\nF1 Score:", f1)

# Visualisasi Model SVM
print("\n=== Visualisasi Model ===")
plt.figure(figsize=(15, 5))

# 1. Visualisasi Training Set
plt.subplot(1, 2, 1)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='winter')
plt.title('Training Set (2 Fitur Pertama)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')

# 2. Visualisasi Test Set
plt.subplot(1, 2, 2)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='winter')
plt.title('Test Set (2 Fitur Pertama)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.tight_layout()
plt.show()

# Heatmap Korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi Antar Fitur')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, hue='Outcome', diag_kind='kde')
plt.show()

# Visualisasi Decision Boundary SVM
from matplotlib.colors import ListedColormap

# Untuk Training Set
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
X_set, y_set = X_train.iloc[:, :2].values, y_train.values
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(6)]).T
pred = clf.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.legend()

# Untuk Test Set
plt.subplot(1, 2, 2)
X_set, y_set = X_test.iloc[:, :2].values, y_test.values
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(6)]).T
pred = clf.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Test set)')
plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.legend()
plt.tight_layout()
plt.show()

# Fungsi untuk memprediksi status diabetes
def prediksi_diabetes(kehamilan, glukosa, tekanan_darah, ketebalan_kulit, insulin, bmi, fungsi_diabetes, umur):
    input_data = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit, 
                           insulin, bmi, fungsi_diabetes, umur]])
    prediksi = clf.predict(input_data)
    return "Diabetes" if prediksi[0] == 1 else "Tidak Diabetes"

# Contoh prediksi
print("\n=== Contoh Prediksi ===")
print("Input sampel:")
print("Jumlah Kehamilan: 6")
print("Glukosa: 148")
print("Tekanan Darah: 72")
print("Ketebalan Kulit: 35")
print("Insulin: 0")
print("BMI: 33.6")
print("Fungsi Diabetes: 0.627")
print("Umur: 50")
print("\nPrediksi:", prediksi_diabetes(6, 148, 72, 35, 0, 33.6, 0.627, 50))


