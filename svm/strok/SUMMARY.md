# 📊 SISTEM PERBANDINGAN MODEL MACHINE LEARNING
## Prediksi Risiko Stroke - Decision Tree vs SVM

### 🎯 **APA YANG TELAH DIBUAT**

Saya telah membuat sistem perbandingan model machine learning yang lengkap untuk prediksi risiko stroke dengan **2 model utama** (Decision Tree dan SVM).

---

## 📁 **FILE YANG DIBUAT**

### 1. **`model_comparison.py`** - Program Utama
- ✅ **2 Model Machine Learning**: Decision Tree dan SVM
- ✅ **Menu Lengkap**: 11 menu dengan fitur komprehensif
- ✅ **Perbandingan Grafik**: Visualisasi perbandingan performa
- ✅ **Evaluasi Komprehensif**: Akurasi, Precision, Recall, F1-Score
- ✅ **Cross-Validation**: Evaluasi yang lebih robust
- ✅ **Prediksi Interaktif**: Input data pasien dan prediksi risiko

### 2. **`README.md`** - Dokumentasi Lengkap
- ✅ **Panduan Penggunaan**: Cara menjalankan program
- ✅ **Penjelasan Menu**: Detail setiap menu dan fungsinya
- ✅ **Parameter Model**: Konfigurasi setiap model
- ✅ **Tips Penggunaan**: Best practices untuk analisis
- ✅ **Troubleshooting**: Solusi untuk masalah umum

### 3. **`demo_run.py`** - Program Demo
- ✅ **Cek Dependencies**: Memastikan semua library terinstall
- ✅ **Cek Dataset**: Memastikan file dataset tersedia
- ✅ **Panduan Interaktif**: Tips dan contoh penggunaan
- ✅ **Auto-launch**: Menjalankan program utama otomatis

### 4. **`example_inputs.txt`** - Contoh Input
- ✅ **Data Pasien Contoh**: 3 contoh pasien dengan risiko berbeda
- ✅ **Mapping Values**: Penjelasan nilai kategorikal
- ✅ **Alur Testing**: Langkah-langkah testing program
- ✅ **Expected Output**: Hasil yang diharapkan

---

## 🚀 **FITUR UTAMA PROGRAM**

### **Model Machine Learning**
1. **Decision Tree** (Gini, max_depth=5)
2. **SVM** (RBF kernel, balanced class weight)

### **Menu Program (11 Menu)**
1. **Muat Dataset** - Load dan preprocess data
2. **Latih Decision Tree** - Training model pohon keputusan
3. **Latih SVM** - Training model support vector machine
4. **Evaluasi Model** - Perbandingan metrik performa
5. **Perbandingan Metrik** - Grafik bar perbandingan
6. **Confusion Matrix** - Matriks kesalahan prediksi
7. **ROC Curve** - Kurva ROC perbandingan
8. **Precision-Recall Curve** - Kurva Precision-Recall
9. **Feature Importance** - Kepentingan fitur
10. **Prediksi** - Input data dan prediksi risiko
11. **Info Dataset** - Informasi dataset

### **Visualisasi Grafik**
- 📊 **Bar Chart**: Perbandingan metrik antar model
- 🎯 **Confusion Matrix**: Heatmap kesalahan prediksi
- 📈 **ROC Curve**: Kurva Receiver Operating Characteristic
- 📉 **Precision-Recall Curve**: Kurva Precision vs Recall
- 🔍 **Feature Importance**: Kepentingan setiap fitur

### **Metrik Evaluasi**
- **Akurasi**: Proporsi prediksi yang benar
- **Precision**: Proporsi prediksi positif yang benar
- **Recall**: Proporsi kasus positif yang terdeteksi
- **F1-Score**: Rata-rata harmonik precision dan recall
- **CV Score**: Cross-validation dengan standar deviasi

---

## 🎯 **CARA MENJALANKAN**

### **Langkah 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Langkah 2: Jalankan Program**
```bash
python model_comparison.py
```

### **Langkah 3: Ikuti Alur Menu**
1. **Menu 1** - Muat dataset
2. **Menu 2,3** - Latih model (Decision Tree dan SVM)
3. **Menu 4** - Evaluasi model
4. **Menu 5-9** - Lihat visualisasi
5. **Menu 10** - Lakukan prediksi
6. **Menu 0** - Keluar

---

## 📊 **PERBANDINGAN MODEL**

### **Decision Tree**
- ✅ **Kelebihan**: Mudah diinterpretasi, cepat training
- ❌ **Kekurangan**: Rentan overfitting, sensitif noise
- 🎯 **Best Use**: Analisis awal, interpretasi aturan

### **SVM**
- ✅ **Kelebihan**: Robust, handle non-linear data
- ❌ **Kekurangan**: Lambat training, sensitif parameter
- 🎯 **Best Use**: Data kompleks, high-dimensional

---

## 🔧 **TECHNICAL SPECIFICATIONS**

### **Dependencies**
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- Python 3.7+

### **Dataset**
- **File**: `dataset/stroke.csv`
- **Fitur**: 10 fitur (gender, age, hypertension, dll.)
- **Target**: stroke (binary classification)
- **Size**: ~5000+ samples

### **Model Parameters**
- **Decision Tree**: Gini, max_depth=5, min_samples_split=10
- **SVM**: RBF kernel, C=1.0, gamma='scale', balanced

---

## 🎉 **KEUNGGULAN SISTEM**

### **1. Komprehensif**
- 2 model machine learning berbeda
- Evaluasi multi-metrik
- Visualisasi lengkap

### **2. User-Friendly**
- Menu interaktif
- Panduan lengkap
- Error handling

### **3. Educational**
- Penjelasan setiap metrik
- Interpretasi hasil
- Best practices

### **4. Practical**
- Prediksi real-time
- Input data pasien
- Output interpretable

---

## 📈 **EXPECTED RESULTS**

### **Performansi Model (Estimasi)**
- **Decision Tree**: Akurasi 85-90%
- **SVM**: Akurasi 88-92%

### **Visualisasi Output**
- Grafik perbandingan metrik
- Confusion matrix heatmap
- ROC dan Precision-Recall curves
- Feature importance bar chart

### **Prediksi Output**
- Risiko stroke (Berisiko/Tidak Berisiko)
- Tingkat kepercayaan (0-1)
- Interpretasi hasil

---

## 🎯 **KESIMPULAN**

Sistem yang dibuat adalah **program perbandingan model machine learning yang lengkap** dengan:

✅ **2 Model Utama**: Decision Tree dan SVM  
✅ **Menu Komprehensif**: 11 menu dengan fitur lengkap  
✅ **Visualisasi Grafik**: Perbandingan dalam bentuk grafik  
✅ **Evaluasi Multi-Metrik**: Akurasi, Precision, Recall, F1-Score  
✅ **Prediksi Interaktif**: Input data dan prediksi real-time  
✅ **Dokumentasi Lengkap**: README, demo, contoh input  

Program ini siap digunakan untuk **analisis perbandingan performa model** dan **prediksi risiko stroke** dengan interface yang user-friendly dan output yang informatif! 🚀 