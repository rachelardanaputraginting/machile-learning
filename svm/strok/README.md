# Sistem Perbandingan Model Machine Learning untuk Prediksi Risiko Stroke

## Deskripsi
Program ini membandingkan performa 2 model machine learning untuk prediksi risiko stroke:
1. **Decision Tree** - Model pohon keputusan dengan kriteria Gini
2. **SVM (Support Vector Machine)** - Model SVM dengan kernel RBF

## Fitur Utama
- ✅ Perbandingan metrik performa (Akurasi, Precision, Recall, F1-Score)
- ✅ Visualisasi grafik perbandingan model
- ✅ Confusion Matrix untuk setiap model
- ✅ ROC Curve dan Precision-Recall Curve
- ✅ Feature Importance (untuk Decision Tree)
- ✅ Prediksi risiko stroke dengan model yang dipilih
- ✅ Cross-validation untuk evaluasi yang lebih robust

## Cara Menjalankan Program

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Program
```bash
python model_comparison.py
```

## Menu Program

### Menu Utama:
1. **Muat dan Proses Dataset** - Memuat dataset stroke.csv
2. **Latih Model Decision Tree** - Melatih model Decision Tree
3. **Latih Model SVM** - Melatih model Support Vector Machine
4. **Evaluasi Semua Model** - Mengevaluasi performa semua model
5. **Tampilkan Perbandingan Metrik (Grafik)** - Grafik perbandingan metrik
6. **Tampilkan Confusion Matrix** - Confusion matrix semua model
7. **Tampilkan ROC Curve** - ROC curve perbandingan
8. **Tampilkan Precision-Recall Curve** - Precision-Recall curve
9. **Tampilkan Feature Importance** - Kepentingan fitur
10. **Lakukan Prediksi** - Prediksi risiko stroke
11. **Tampilkan Informasi Dataset** - Informasi dataset
0. **Keluar** - Keluar dari program

## Alur Penggunaan

### Langkah 1: Muat Dataset
Pilih menu **1** untuk memuat dan memproses dataset stroke.csv

### Langkah 2: Latih Model
Pilih menu **2, 3** untuk melatih model-model yang diinginkan:
- Menu 2: Decision Tree
- Menu 3: SVM

### Langkah 3: Evaluasi Model
Pilih menu **4** untuk mengevaluasi semua model yang telah dilatih

### Langkah 4: Analisis Visual
Pilih menu **5-9** untuk melihat berbagai visualisasi perbandingan:
- Menu 5: Perbandingan metrik dalam grafik bar
- Menu 6: Confusion matrix semua model
- Menu 7: ROC curve perbandingan
- Menu 8: Precision-Recall curve
- Menu 9: Feature importance

### Langkah 5: Prediksi
Pilih menu **10** untuk melakukan prediksi risiko stroke dengan model yang dipilih

## Output Program

### 1. Metrik Evaluasi
- **Akurasi**: Proporsi prediksi yang benar
- **Precision**: Proporsi prediksi positif yang benar
- **Recall**: Proporsi kasus positif yang terdeteksi
- **F1-Score**: Rata-rata harmonik precision dan recall
- **CV Score**: Cross-validation score dengan standar deviasi

### 2. Visualisasi
- **Grafik Bar**: Perbandingan metrik antar model
- **Confusion Matrix**: Matriks kesalahan prediksi
- **ROC Curve**: Kurva Receiver Operating Characteristic
- **Precision-Recall Curve**: Kurva Precision vs Recall
- **Feature Importance**: Kepentingan setiap fitur

### 3. Prediksi
- Input data pasien (umur, gender, BMI, dll.)
- Output: Risiko stroke (Berisiko/Tidak Berisiko)
- Tingkat kepercayaan prediksi

## Dataset
Program menggunakan dataset `stroke.csv` yang berisi:
- **Fitur**: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- **Target**: stroke (0: Tidak stroke, 1: Stroke)

## Parameter Model

### Decision Tree
- criterion: 'gini'
- max_depth: 5
- min_samples_split: 10
- min_samples_leaf: 5

### SVM
- kernel: 'rbf'
- C: 1.0
- gamma: 'scale'
- class_weight: 'balanced'

## Tips Penggunaan
1. **Selalu muat dataset terlebih dahulu** (menu 1)
2. **Latih kedua model** untuk perbandingan yang berarti
3. **Evaluasi model** sebelum melihat visualisasi
4. **Gunakan cross-validation score** untuk evaluasi yang lebih akurat
5. **Perhatikan feature importance** untuk memahami faktor risiko utama

## Troubleshooting
- **Error "No module named 'pandas'"**: Install dependencies dengan `pip install -r requirements.txt`
- **Error "File not found"**: Pastikan file `dataset/stroke.csv` ada di folder yang benar
- **Error visualisasi**: Pastikan matplotlib dan seaborn terinstall dengan benar 