import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

def train_sample_models():
    """
    Script contoh untuk melatih dan menyimpa model ML klasifikasi:
    - Random Forest (rf_model.pkl)
    - SVM (svm_model.pkl)
    - Logistic Regression (logreg_model.pkl)
    - KNN (knn_model.pkl)
    """
    print("Memuat model embedding LaBSE...")
    labse_embedding = SentenceTransformer('sentence-transformers/LaBSE')

    # Data contoh (ganti dengan dataset Anda)
    texts = [
        "Bagaimana cara menghitung luas lingkaran?",
        "Rumus pythagoras dan phytagoras segitiga siku siku",
        "Pemrograman dasar python dan struktur data array",
        "Cara merakit PC gaming budget 5 juta dan install OS",
        "Sistem pencernaan manusia dan fungsi organ lambung",
        "Proses fotosintesis pada tumbuhan hijau"
    ]

    labels = [
        "Matematika - SD",
        "Matematika - SMP",
        "Informatika - SMA",
        "Informatika - SMK",
        "Biologi - SMP",
        "Biologi - SMA"
    ]

    print("Membangun embeddings data latih...")
    X = labse_embedding.encode(texts)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "src", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Melatih model Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))

    print("Melatih model SVM...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X, y)
    joblib.dump(svm, os.path.join(models_dir, "svm_model.pkl"))

    print("Melatih model Logistic Regression...")
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X, y)
    joblib.dump(logreg, os.path.join(models_dir, "logreg_model.pkl"))

    print("Melatih model KNN...")
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, y)
    joblib.dump(knn, os.path.join(models_dir, "knn_model.pkl"))

    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

    print("✅ Semua model ML berhasil dilatih dan disimpan di src/models/")

if __name__ == "__main__":
    train_sample_models()
