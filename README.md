# Eksperimen Machine Learning: Telco Customer Churn Prediction

## 📋 Informasi Proyek

**Nama:** SHAH FIRIZKI AZMI  
**Kelas:** Sistem Monitoring dan Logging Machine Learning (SMSML)  
**Dataset:** Telco Customer Churn  
**Task:** Binary Classification (Prediksi Churn Pelanggan)  

## 🎯 Tujuan Proyek

Membangun sistem machine learning end-to-end yang lengkap untuk memprediksi customer churn pada perusahaan telekomunikasi, dengan implementasi monitoring dan logging yang comprehensive menggunakan MLflow, Prometheus, dan Grafana.

## 🏗️ Arsitektur Sistem

Proyek ini dibangun dengan prinsip **Clean Code** yang mengutamakan:

### 🔹 1. **Scalability**
- Desain modular yang dapat berkembang sesuai kebutuhan
- Struktur folder yang terorganisir dengan baik
- Konfigurasi yang dapat disesuaikan untuk berbagai environment

### 🔹 2. **Readability**
- Kode yang mudah dibaca dengan naming convention yang konsisten
- Dokumentasi lengkap pada setiap modul
- Komentar yang informatif dan tidak berlebihan

### 🔹 3. **Maintainability**
- Implementasi prinsip DRY (Don't Repeat Yourself)
- Separation of concerns yang jelas
- Error handling yang comprehensive

### 🔹 4. **Reusability**
- Fungsi dan kelas yang dapat digunakan ulang
- Abstraksi yang baik untuk preprocessing dan modeling
- Template yang dapat diadaptasi untuk dataset lain

### 🔹 5. **Performance**
- Optimasi preprocessing dengan pandas dan numpy
- Caching untuk operasi yang berulang
- Monitoring performa real-time

### 🔹 6. **Testability**
- Struktur kode yang memudahkan unit testing
- Validation pipeline yang robust
- Health check untuk setiap komponen

### 🔹 7. **Security**
- Validasi input yang ketat
- Environment variable untuk konfigurasi sensitif
- Sanitasi data untuk mencegah injection

### 🔹 8. **Portability**
- Docker support (ready)
- Environment configuration yang fleksibel
- Cross-platform compatibility

### 🔹 9. **Documentation**
- README lengkap dengan contoh penggunaan
- Docstring pada setiap fungsi
- API documentation

## 📁 Struktur Proyek

```
Eksperimen_SML_SHAH-FIRIZKI-AZMI/
├── 📊 WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 📓 Template_Eksperimen_MSML.ipynb
├── 📂 preprocessing/
│   ├── 🐍 automate_SHAH-FIRIZKI-AZMI.py
│   ├── 📈 telco_churn_preprocessing.csv
│   ├── 🔧 scaler.pkl
│   ├── 🏷️ label_encoders.pkl
│   ├── 📋 feature_names.pkl
│   └── ⚙️ preprocessing_config.pkl
├── 📂 Membangun_model/
│   ├── 🤖 modelling.py
│   ├── 🎯 modelling_tuning.py
│   ├── 📈 telco_churn_preprocessing.csv
│   ├── 📄 requirements.txt
│   ├── 🖼️ screenshoot_dashboard.jpg (akan dibuat)
│   ├── 🖼️ screenshoot_artifak.jpg (akan dibuat)
│   └── 🔗 DagsHub.txt (optional untuk advanced)
├── 📂 .github/workflows/
│   └── 🔄 preprocessing.yml
├── 📂 Monitoring dan Logging/
│   ├── 📊 2.prometheus.yml
│   ├── 📈 3.prometheus_exporter.py
│   ├── 🔮 7.inference.py
│   ├── 📂 4.bukti monitoring Prometheus/
│   ├── 📂 5.bukti monitoring Grafana/
│   └── 📂 6.bukti alerting Grafana/
└── 📖 README.md
```

## 🚀 Quick Start Guide

### 1. **Prerequisites**
```bash
# Python 3.12.7
# MLflow 2.19.0
# Docker (optional)
# Grafana
# Prometheus
```

### 2. **Installation**
```bash
# Clone repository
git clone https://github.com/username/Eksperimen_SML_SHAH-FIRIZKI-AZMI.git
cd Eksperimen_SML_SHAH-FIRIZKI-AZMI

# Install dependencies
pip install -r Membangun_model/requirements.txt
```

### 3. **Data Preprocessing**
```bash
# Automated preprocessing
python preprocessing/automate_SHAH-FIRIZKI-AZMI.py

# Output: preprocessed data + artifacts saved
```

### 4. **Model Training**
```bash
cd Membangun_model

# Basic training dengan MLflow autolog
python modelling.py

# Advanced training dengan hyperparameter tuning
python modelling_tuning.py
```

### 5. **Start MLflow Server**
```bash
mlflow server --host 127.0.0.1 --port 5000
# Access: http://localhost:5000
```

### 6. **Model Serving**
```bash
# Serve model (setelah training)
mlflow models serve -m "runs:/<RUN_ID>/model" -p 1234 --no-conda
```

### 7. **Monitoring Setup**
```bash
cd "Monitoring dan Logging"

# Start Prometheus metrics exporter
python 3.prometheus_exporter.py

# Start Prometheus (setelah install)
prometheus --config.file=2.prometheus.yml

# Start Grafana (setelah install)
# Create dashboard dengan nama username Dicoding
```

## 🔧 Komponen Utama

### 📊 **Data Preprocessing** (`preprocessing/`)
- **Otomatisasi lengkap** preprocessing pipeline
- **Error handling** yang robust
- **Artifacts management** untuk reproducibility
- **Configurability** untuk berbagai environment

**Key Features:**
- Missing value handling
- Categorical encoding dengan LabelEncoder
- Feature scaling dengan StandardScaler
- Data validation dan quality checks

### 🤖 **Model Training** (`Membangun_model/`)
- **MLflow integration** untuk experiment tracking
- **Hyperparameter tuning** dengan GridSearch/RandomizedSearch
- **Manual logging** dengan comprehensive metrics
- **Model versioning** dan artifact management

**Supported Models:**
- Random Forest Classifier
- Gradient Boosting Classifier
- Extensible untuk model lain

### 📈 **Monitoring & Logging** (`Monitoring dan Logging/`)
- **Prometheus metrics** collection
- **Grafana dashboard** untuk visualisasi
- **Real-time monitoring** system dan model performance
- **Alerting system** untuk anomaly detection

**Metrics yang dipantau:**
- System metrics (CPU, Memory, Disk, Network)
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Application metrics (Response time, Throughput, Error rate)
- Business metrics (Prediction distribution, Model drift)

### 🔄 **CI/CD Pipeline** (`.github/workflows/`)
- **Automated preprocessing** pada setiap push
- **Artifact management** dengan GitHub Actions
- **Quality validation** pipeline
- **Deployment ready** configuration

## 📋 Kriteria Submission

### ✅ **Kriteria 1: Melakukan Eksperimen terhadap Dataset Pelatihan**
- [x] **Basic (2 pts):** Notebook eksperimen lengkap dengan EDA, preprocessing, modeling
- [x] **Skilled (3 pts):** File automatisasi `automate_SHAH-FIRIZKI-AZMI.py`
- [x] **Advance (4 pts):** GitHub Actions workflow untuk preprocessing automation

### ✅ **Kriteria 2: Membangun Model Machine Learning**
- [x] **Basic (2 pts):** MLflow tracking dengan autolog (`modelling.py`)
- [x] **Skilled (3 pts):** Hyperparameter tuning dengan manual logging (`modelling_tuning.py`)
- [ ] **Advance (4 pts):** DagsHub integration (konfigurasi ready, perlu setup account)

### ✅ **Kriteria 3: Membuat Workflow CI**
- [x] **Basic (2 pts):** GitHub Actions workflow
- [ ] **Skilled (3 pts):** Artifact upload ke repository
- [ ] **Advance (4 pts):** Docker image building

### ✅ **Kriteria 4: Membuat Sistem Monitoring dan Logging**
- [x] **Basic (2 pts):** Prometheus + Grafana dengan 3+ metrics
- [x] **Skilled (3 pts):** 5+ metrics + 1 alerting rule
- [x] **Advance (4 pts):** 10+ metrics + 3 alerting rules

## 🎯 **Langkah-langkah Eksekusi**

### **Fase 1: Development**
1. ✅ Setup repository dan struktur folder
2. ✅ Implementasi preprocessing automation
3. ✅ Implementasi model training dengan MLflow
4. ✅ Setup monitoring dan logging system

### **Fase 2: Deployment**
1. ⏳ Setup MLflow server
2. ⏳ Training model dan screenshot artifacts
3. ⏳ Setup Prometheus dan Grafana
4. ⏳ Konfigurasi dashboard dan alerting

### **Fase 3: Documentation**
1. ⏳ Capture screenshots untuk submission
2. ⏳ Create demo video (optional)
3. ⏳ Final testing dan validation

## 📸 Screenshot Requirements

### **MLflow Screenshots:**
- [ ] Dashboard showing experiments
- [ ] Model artifacts dan metrics
- [ ] Model serving interface

### **Prometheus Screenshots:**
- [ ] Metrics collection (minimum 3 untuk Basic)
- [ ] Real-time monitoring interface

### **Grafana Screenshots:**
- [ ] Dashboard dengan nama username Dicoding
- [ ] Visualisasi metrics (minimum sesuai level)
- [ ] Alerting rules dan notifications

## 🛠️ **Troubleshooting**

### **Common Issues:**
1. **MLflow server tidak start:** Check port 5000 availability
2. **Model serving error:** Ensure model trained dan run_id valid
3. **Prometheus metrics tidak muncul:** Check exporter running di port 8000
4. **Grafana connection issue:** Verify Prometheus data source

### **Performance Tips:**
1. Use `low_memory=False` untuk large datasets
2. Implement caching untuk repeated operations  
3. Monitor memory usage selama training
4. Use background processes untuk metrics collection

## 🤝 **Contributing**

Proyek ini mengikuti prinsip clean code dan best practices. Untuk kontribusi:

1. Fork repository
2. Create feature branch
3. Implement dengan testing
4. Submit pull request dengan dokumentasi

## 📞 **Support**

Untuk pertanyaan dan support:
- **Email:** [Your email]
- **GitHub Issues:** [Repository issues]
- **Documentation:** Lihat docstring dalam kode

## 🏆 **Achievement Goals**

- [ ] **Basic Level:** 2 pts pada setiap kriteria ✅
- [ ] **Skilled Level:** 3 pts pada setiap kriteria ⏳
- [ ] **Advanced Level:** 4 pts pada setiap kriteria ⏳

## 📝 **License**

Proyek ini dibuat untuk keperluan submission kelas SMSML Dicoding.

---

**Dibuat dengan ❤️ menggunakan Clean Code Principles**  
**SHAH FIRIZKI AZMI - 2025**