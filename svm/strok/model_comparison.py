# === Import library yang dibutuhkan ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, 
                           precision_score, recall_score, classification_report,
                           roc_curve, auc, precision_recall_curve)
import warnings
from io import StringIO
try:
    import pydotplus
except ImportError:
    pydotplus = None

# === Inisialisasi variabel global ===
# Model-model yang akan digunakan
dt_model = None          # Decision Tree model
svm_model = None         # SVM model
dt_model_optimized = None # Decision Tree model yang dioptimalkan

# Data yang akan digunakan
X_train = X_test = y_train = y_test = None
X_scaled_train = X_scaled_test = None
scaler = None

# Hasil evaluasi model
model_results = {}

# Daftar fitur yang akan digunakan
feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
 
# === Fungsi untuk memuat dan memproses dataset ===
def load_and_preprocess_data():
    """Memuat dan memproses dataset stroke"""
    global X_train, X_test, y_train, y_test, X_scaled_train, X_scaled_test, scaler
    
    print("=== MEMUAT DAN MEMPROSES DATASET ===")
    
    try:
        # Baca file CSV
        df = pd.read_csv("dataset/stroke.csv")
        print(f"Dataset berhasil dimuat dengan {df.shape[0]} baris dan {df.shape[1]} kolom")
        
        # Hapus kolom id
        df = df.drop('id', axis=1)
        
        # Handle missing values
        df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
        
        # Konversi data kategorikal ke numerik
        le = LabelEncoder()
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for column in categorical_columns:
            df[column] = le.fit_transform(df[column])
        
        # Pisahkan fitur dan target
        X = df[feature_cols]
        y = df['stroke']
        
        # Bagi data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardisasi data untuk SVM
        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_train)
        X_scaled_test = scaler.transform(X_test)
        
        print("Dataset berhasil diproses!")
        print(f"Data latih: {X_train.shape[0]} sampel")
        print(f"Data uji: {X_test.shape[0]} sampel")
        print(f"Fitur yang digunakan: {len(feature_cols)} fitur")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error dalam memuat dataset: {str(e)}")
        return False

# === Fungsi untuk melatih model Decision Tree ===
def train_decision_tree():
    """Melatih model Decision Tree dengan berbagai parameter"""
    global dt_model
    
    print("=== MELATIH MODEL DECISION TREE (GINI) ===")
    
    # Model Decision Tree dengan Gini
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    print("Model Decision Tree (Gini) berhasil dilatih!")
    print(f"Parameter: criterion='gini', max_depth=3")
    print()

# === Fungsi untuk melatih model Decision Tree yang dioptimalkan (Entropy) ===
def train_optimized_decision_tree():
    """Melatih model Decision Tree yang dioptimalkan (Entropy)."""
    global dt_model_optimized
    
    print("=== MELATIH MODEL DECISION TREE (OPTIMIZED) ===")
    
    # Model Decision Tree dengan Entropy
    dt_model_optimized = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42
    )
    
    dt_model_optimized.fit(X_train, y_train)
    print("Model Decision Tree (Optimized/Entropy) berhasil dilatih!")
    print(f"Parameter: criterion='entropy', max_depth=3")
    print()

# === Fungsi untuk melatih model SVM ===
def train_svm():
    """Melatih model SVM"""
    global svm_model
    
    print("=== MELATIH MODEL SVM ===")
    
    # Model SVM dengan RBF kernel
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    svm_model.fit(X_scaled_train, y_train)
    print("Model SVM berhasil dilatih!")
    print(f"Parameter: kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'")
    print()

# === Fungsi untuk mengevaluasi model ===
def evaluate_model(model, model_name, X_test_data, y_test_data):
    """Mengevaluasi model dan mengembalikan metrik"""
    
    # Prediksi
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Hitung metrik
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, zero_division=0)
    recall = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    
    # Cross validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Simpan hasil
    model_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return model_results[model_name]

# === Fungsi untuk mengevaluasi semua model ===
def evaluate_all_models():
    """Mengevaluasi semua model yang telah dilatih pada data uji."""
    
    print("=== EVALUASI SEMUA MODEL PADA DATA UJI ===")
    
    if dt_model is not None:
        print("\n--- Decision Tree (Gini) ---")
        results_dt = evaluate_model(dt_model, 'Decision Tree (Gini)', X_test, y_test)
        print(f"Akurasi: {results_dt['accuracy']:.4f}")
        print(f"Precision: {results_dt['precision']:.4f}")
        print(f"Recall: {results_dt['recall']:.4f}")
        print(f"F1-Score: {results_dt['f1_score']:.4f}")
        print(f"CV Score (pada data latih): {results_dt['cv_mean']:.4f} (+/- {results_dt['cv_std']*2:.4f})")
    
    if dt_model_optimized is not None:
        print("\n--- Decision Tree (Optimized/Entropy) ---")
        results_dt_opt = evaluate_model(dt_model_optimized, 'Decision Tree (Entropy)', X_test, y_test)
        print(f"Akurasi: {results_dt_opt['accuracy']:.4f}")
        print(f"Precision: {results_dt_opt['precision']:.4f}")
        print(f"Recall: {results_dt_opt['recall']:.4f}")
        print(f"F1-Score: {results_dt_opt['f1_score']:.4f}")
        print(f"CV Score (pada data latih): {results_dt_opt['cv_mean']:.4f} (+/- {results_dt_opt['cv_std']*2:.4f})")

    if svm_model is not None:
        print("\n--- SVM ---")
        results_svm = evaluate_model(svm_model, 'SVM', X_scaled_test, y_test)
        print(f"Akurasi: {results_svm['accuracy']:.4f}")
        print(f"Precision: {results_svm['precision']:.4f}")
        print(f"Recall: {results_svm['recall']:.4f}")
        print(f"F1-Score: {results_svm['f1_score']:.4f}")
        print(f"CV Score (pada data latih): {results_svm['cv_mean']:.4f} (+/- {results_svm['cv_std']*2:.4f})")
    
    print()

# === Fungsi untuk menampilkan perbandingan metrik ===
def show_metrics_comparison():
    """Menampilkan perbandingan metrik dalam bentuk grafik"""
    
    if not model_results:
        print("Belum ada model yang dievaluasi!")
        return
    
    print("=== PERBANDINGAN METRIK MODEL ===")
    
    # Siapkan data untuk plotting
    models = list(model_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Buat subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Perbandingan Metrik Model Decision Tree vs SVM', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics_names):
        row = i // 2
        col = i % 2
        
        values = [model_results[model][metric] for model in models]
        
        bars = axes[row, col].bar(models, values, color=['skyblue', 'lightcoral'][:len(models)])
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_ylabel('Score')
        axes[row, col].set_ylim(0, 1)
        
        # Tambahkan nilai di atas bar
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# === Fungsi untuk menampilkan confusion matrix ===
def show_confusion_matrices():
    """Menampilkan confusion matrix untuk semua model"""
    
    if not model_results:
        print("Belum ada model yang dievaluasi!")
        return
    
    print("=== CONFUSION MATRIX SEMUA MODEL ===")
    
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(model_results.items()):
        cm = confusion_matrix(y_test, results['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {model_name}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

# === Fungsi untuk menampilkan Precision-Recall Curve ===
def show_precision_recall_curves():
    """Menampilkan Precision-Recall curve untuk semua model"""
    
    if not model_results:
        print("Belum ada model yang dievaluasi!")
        return
    
    print("=== PRECISION-RECALL CURVE SEMUA MODEL ===")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, results in model_results.items():
        if results['y_pred_proba'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            
            plt.plot(recall, precision, label=f'{model_name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison - Decision Tree vs SVM')
    plt.legend()
    plt.grid(True)
    plt.show()

# === Fungsi untuk menampilkan feature importance (untuk Decision Tree) ===
def show_feature_importance():
    """Menampilkan feature importance untuk model Decision Tree"""
    
    print("=== FEATURE IMPORTANCE ===")
    
    if dt_model is None:
        print("Model Decision Tree belum dilatih!")
        return
    
    # Ambil feature importance dari Decision Tree
    importance = dt_model.feature_importances_
    
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.title('Feature Importance - Decision Tree')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# === Fungsi untuk menampilkan visualisasi pohon keputusan ===
def visualize_decision_tree():
    """Menyimpan visualisasi salah satu model Decision Tree sebagai gambar JPG."""
    print("=== VISUALISASI POHON KEPUTUSAN ===")
    
    models_to_visualize = {}
    if dt_model:
        models_to_visualize['1'] = ("Default (Gini)", dt_model)
    if dt_model_optimized:
        models_to_visualize['2'] = ("Optimized (Entropy)", dt_model_optimized)

    if not models_to_visualize:
        print("Tidak ada model Decision Tree yang sudah dilatih!")
        return

    print("Pilih model pohon keputusan yang ingin divisualisasikan:")
    for key, (name, _) in models_to_visualize.items():
        print(f"{key}. {name}")
    
    choice = input("Pilihan: ")
    if choice not in models_to_visualize:
        print("Pilihan tidak valid.")
        return

    model_name, selected_model = models_to_visualize[choice]
    file_name = f"decision_tree_{model_name.split(' ')[0].lower()}.jpg"
    
    if pydotplus is None:
        print("❌ Library 'pydotplus' tidak ditemukan. Harap install terlebih dahulu.")
        print("   Jalankan: pip install pydotplus")
        return

    try:
        dot_data = StringIO()
        export_graphviz(selected_model, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names=feature_cols,
                        class_names=['Tidak Stroke', 'Stroke'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # Mengubah output ke format JPG
        graph.write_jpg(file_name)
        print(f"✅ Visualisasi Decision Tree berhasil disimpan sebagai '{file_name}'")
        print("   Anda bisa membuka file gambar tersebut di folder yang sama dengan skrip ini.")
    except Exception as e:
        print(f"❌ Error saat membuat visualisasi: {str(e)}")
        print("\n--- SOLUSI ---")
        print("ℹ️ Error ini biasanya terjadi karena program Graphviz tidak ditemukan di sistem Anda, meskipun 'pydotplus' sudah ter-install.")
        print("1. **Install Graphviz**: Download dan install dari situs resminya: https://graphviz.org/download/")
        print("2. **PENTING - Tambahkan ke PATH**: Saat instalasi, PASTIKAN Anda mencentang opsi seperti 'Add Graphviz to the system PATH'.")
        print("3. **Restart Terminal/IDE**: Setelah instalasi, tutup dan buka kembali terminal atau IDE Anda agar PATH yang baru terbaca.")

# === Fungsi untuk mengekspor hasil ke file CSV ===
def export_results_to_csv():
    """Mengekspor hasil perbandingan metrik dan prediksi ke file CSV."""
    if not model_results:
        print("Belum ada model yang dievaluasi! Jalankan menu evaluasi terlebih dahulu.")
        return

    print("=== EKSPOR HASIL KE CSV ===")

    # 1. Ekspor perbandingan metrik
    try:
        metrics_df = pd.DataFrame(model_results).transpose()
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']]
        metrics_df.to_csv('model_comparison_results.csv')
        print("✅ Hasil perbandingan metrik disimpan ke 'model_comparison_results.csv'")
    except Exception as e:
        print(f"❌ Gagal menyimpan perbandingan metrik: {e}")

    # 2. Ekspor prediksi Decision Tree
    if 'Decision Tree (Gini)' in model_results:
        try:
            dt_results_df = X_test.copy()
            dt_results_df['actual_stroke'] = y_test.values
            dt_results_df['predicted_stroke_dt_gini'] = model_results['Decision Tree (Gini)']['y_pred']
            dt_results_df.to_csv('decision_tree_gini_predictions.csv', index=False)
            print("✅ Hasil prediksi Decision Tree (Gini) disimpan ke 'decision_tree_gini_predictions.csv'")
        except Exception as e:
            print(f"❌ Gagal menyimpan prediksi Decision Tree (Gini): {e}")

    # 3. Ekspor prediksi Decision Tree (Optimized)
    if 'Decision Tree (Entropy)' in model_results:
        try:
            dt_opt_results_df = X_test.copy()
            dt_opt_results_df['actual_stroke'] = y_test.values
            dt_opt_results_df['predicted_stroke_dt_entropy'] = model_results['Decision Tree (Entropy)']['y_pred']
            dt_opt_results_df.to_csv('decision_tree_entropy_predictions.csv', index=False)
            print("✅ Hasil prediksi Decision Tree (Entropy) disimpan ke 'decision_tree_entropy_predictions.csv'")
        except Exception as e:
            print(f"❌ Gagal menyimpan prediksi Decision Tree (Entropy): {e}")

    # 4. Ekspor prediksi SVM
    if 'SVM' in model_results:
        try:
            svm_results_df = X_test.copy()
            svm_results_df['actual_stroke'] = y_test.values
            svm_results_df['predicted_stroke_svm'] = model_results['SVM']['y_pred']
            svm_results_df.to_csv('svm_predictions.csv', index=False)
            print("✅ Hasil prediksi SVM disimpan ke 'svm_predictions.csv'")
        except Exception as e:
            print(f"❌ Gagal menyimpan prediksi SVM: {e}")
    
    print()

# === Fungsi untuk menampilkan menu utama ===
def show_menu():
    """Menampilkan menu utama"""
    print("\n" + "="*60)
    print("           SISTEM PERBANDINGAN MODEL MACHINE LEARNING")
    print("                    PREDIKSI RISIKO STROKE")
    print("="*60)
    print("1. Muat dan Proses Dataset")
    print("2. Latih Model Decision Tree (Gini)")
    print("3. Latih Model Decision Tree (Entropy)")
    print("4. Latih Model SVM")
    print("5. Evaluasi Semua Model")
    print("--- VISUALISASI ---")
    print("6. Tampilkan Perbandingan Metrik (Grafik)")
    print("7. Tampilkan Confusion Matrix")
    print("8. Tampilkan Precision-Recall Curve")
    print("9. Tampilkan Feature Importance (DT)")
    print("10. Tampilkan Visualisasi Pohon Keputusan (Gambar)")
    print("--- EKSPOR & PREDIKSI ---")
    print("11. Ekspor Hasil ke CSV")
    print("12. Lakukan Prediksi")
    print("--- LAINNYA ---")
    print("13. Tampilkan Informasi Dataset")
    print("0. Keluar")
    print("="*60)

# === Fungsi untuk menampilkan informasi dataset ===
def show_dataset_info():
    """Menampilkan informasi tentang dataset"""
    
    try:
        df = pd.read_csv("dataset/stroke.csv")
        
        print("=== INFORMASI DATASET ===")
        print(f"Jumlah baris: {df.shape[0]}")
        print(f"Jumlah kolom: {df.shape[1]}")
        print(f"Kolom: {list(df.columns)}")
        print("\nStatistik deskriptif:")
        print(df.describe())
        print("\nInformasi tipe data:")
        print(df.info())
        print("\nJumlah missing values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error dalam membaca dataset: {str(e)}")

# === Fungsi utama ===
def main():
    """Fungsi utama program"""
    
    print("Selamat datang di Sistem Perbandingan Model Machine Learning!")
    print("Program ini akan membandingkan performa Decision Tree dan SVM")
    print("untuk prediksi risiko stroke.")
    
    while True:
        show_menu()
        choice = input("\nPilih menu (0-13): ")
        
        if choice == '1':
            load_and_preprocess_data()
        elif choice == '2':
            if X_train is None:
                print("Harap muat dataset terlebih dahulu (menu 1)!")
            else:
                train_decision_tree()
        elif choice == '3':
            if X_train is None:
                print("Harap muat dataset terlebih dahulu (menu 1)!")
            else:
                train_optimized_decision_tree()
        elif choice == '4':
            if X_train is None:
                print("Harap muat dataset terlebih dahulu (menu 1)!")
            else:
                train_svm()
        elif choice == '5':
            if not any([dt_model, svm_model, dt_model_optimized]):
                print("Harap latih setidaknya satu model terlebih dahulu!")
            else:
                evaluate_all_models()
        elif choice == '6':
            if not model_results:
                print("Harap evaluasi model terlebih dahulu (menu 5)!")
            else:
                show_metrics_comparison()
        elif choice == '7':
            if not model_results:
                print("Harap evaluasi model terlebih dahulu (menu 5)!")
            else:
                show_confusion_matrices()
        elif choice == '8':
            if not model_results:
                print("Harap evaluasi model terlebih dahulu (menu 5)!")
            else:
                show_precision_recall_curves()
        elif choice == '9':
            if not any([dt_model, dt_model_optimized]):
                print("Harap latih model Decision Tree terlebih dahulu!")
            else:
                show_feature_importance()
        elif choice == '10':
            visualize_decision_tree()
        elif choice == '11':
            export_results_to_csv()
        elif choice == '12':
            if not model_results:
                print("Harap latih dan evaluasi model terlebih dahulu!")
            else:
                make_prediction_with_model()
        elif choice == '13':
            show_dataset_info()
        elif choice == '0':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        else:
            print("Pilihan tidak valid! Silakan pilih 0-13.")
        
        input("\nTekan Enter untuk melanjutkan...")

# Jalankan program jika file dijalankan langsung
if __name__ == "__main__":
    main() 