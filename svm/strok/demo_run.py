# Demo Program Perbandingan Model Machine Learning
# File ini menunjukkan cara menjalankan program model_comparison.py

import subprocess
import sys
import os

def check_dependencies():
    """Mengecek apakah semua dependencies terinstall"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Dependencies yang hilang:", missing_packages)
        print("📦 Install dengan perintah: pip install -r requirements.txt")
        return False
    
    print("✅ Semua dependencies terinstall!")
    return True

def check_dataset():
    """Mengecek apakah dataset tersedia"""
    dataset_path = "dataset/stroke.csv"
    
    if os.path.exists(dataset_path):
        print("✅ Dataset stroke.csv ditemukan!")
        return True
    else:
        print("❌ Dataset stroke.csv tidak ditemukan!")
        print("📁 Pastikan file ada di folder dataset/")
        return False

def run_demo():
    """Menjalankan demo program"""
    print("🚀 DEMO PROGRAM PERBANDINGAN MODEL MACHINE LEARNING")
    print("=" * 60)
    
    # Cek dependencies
    print("\n1. Mengecek dependencies...")
    if not check_dependencies():
        return
    
    # Cek dataset
    print("\n2. Mengecek dataset...")
    if not check_dataset():
        return
    
    print("\n3. Menjalankan program utama...")
    print("📋 Menu yang tersedia:")
    print("   1. Muat dan Proses Dataset")
    print("   2. Latih Model Decision Tree")
    print("   3. Latih Model SVM")
    print("   4. Evaluasi Semua Model")
    print("   5. Tampilkan Perbandingan Metrik (Grafik)")
    print("   6. Tampilkan Confusion Matrix")
    print("   7. Tampilkan ROC Curve")
    print("   8. Tampilkan Precision-Recall Curve")
    print("   9. Tampilkan Feature Importance")
    print("   10. Lakukan Prediksi")
    print("   11. Tampilkan Informasi Dataset")
    print("   0. Keluar")
    
    print("\n💡 Tips penggunaan:")
    print("   - Jalankan menu secara berurutan: 1 → 2,3 → 4 → 5-9")
    print("   - Menu 1 wajib dijalankan terlebih dahulu")
    print("   - Latih kedua model untuk perbandingan yang berarti")
    print("   - Evaluasi model (menu 4) sebelum melihat visualisasi")
    
    print("\n🎯 Contoh alur penggunaan:")
    print("   1. Pilih menu 1 (Muat Dataset)")
    print("   2. Pilih menu 2 (Latih Decision Tree)")
    print("   3. Pilih menu 3 (Latih SVM)")
    print("   4. Pilih menu 4 (Evaluasi Model)")
    print("   5. Pilih menu 5 (Perbandingan Metrik)")
    print("   6. Pilih menu 10 (Prediksi)")
    
    print("\n🚀 Memulai program...")
    print("=" * 60)
    
    try:
        # Jalankan program utama
        subprocess.run([sys.executable, "model_comparison.py"])
    except KeyboardInterrupt:
        print("\n\n⏹️ Program dihentikan oleh user")
    except Exception as e:
        print(f"\n❌ Error menjalankan program: {str(e)}")

if __name__ == "__main__":
    run_demo() 