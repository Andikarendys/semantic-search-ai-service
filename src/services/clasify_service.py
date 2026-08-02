import os
import re
import unicodedata
import joblib
import numpy as np
from src.embed_model import get_embedding_model

def get_labse_embedding():
    return get_embedding_model()

MODEL_DIR = "src/models"

OPTIMAL_THRESHOLD_PER_ALGO = {
    "knn": 0.59,
    "svm": 0.37,
    "logreg": 0.52,
    "rf": 0.09,
}
DEFAULT_ALGORITHM = "svm"

MODEL_FILENAMES = {
    "knn": ["model_knn.pkl", "knn_model.pkl"],
    "svm": ["model_svm.pkl", "svm_model.pkl"],
    "logreg": ["model_logreg.pkl", "logreg_model.pkl"],
    "rf": ["model_randomforest.pkl", "rf_model.pkl"],
}

ALGO_DISPLAY_NAME = {
    "rf": "Random forest",
    "svm": "SVM",
    "logreg": "Logistic regression",
    "knn": "KNN",
}

ALGORITHM_WARNINGS = {
    "rf": ("Random Forest memiliki akurasi relatif lebih rendah dibanding "
           "algoritma lain pada pengujian model (F1-score ±0.59 vs ±0.65-0.69).")
}

# ── Lazy-Load Resources ──────────────────────────────────────────────
_labse_embedding = None
_label_encoder = None
_models = {}

def get_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        if os.path.exists(path):
            _label_encoder = joblib.load(path)
    return _label_encoder

def get_model(algo_key: str):
    global _models
    if algo_key not in _models:
        filenames = MODEL_FILENAMES.get(algo_key, [])
        for fname in filenames:
            fpath = os.path.join(MODEL_DIR, fname)
            if os.path.exists(fpath):
                try:
                    _models[algo_key] = joblib.load(fpath)
                    break
                except Exception as e:
                    print(f"[WARNING] Gagal memuat model '{fname}': {e}")
    return _models.get(algo_key)

def preservation_clean(text: str) -> str:
    """
    Normalisasi & pembersihan teks persis sesuai notebook training.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip('"').strip("'")
    mojibake = {
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': '-', 'Â': '',
        '\r\n': ' ', '\r': ' ', '\n': ' ',
    }
    for bad, good in mojibake.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def _predict_one(clean_text: str, algorithm: str, threshold: float = None) -> dict:
    model = get_model(algorithm)
    if not model:
        raise ValueError(f"Algoritma '{algorithm}' tidak tersedia / gagal dimuat.")

    label_enc = get_label_encoder()
    vec = get_labse_embedding().encode([clean_text])

    # Ambil probabilitas
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / np.sum(exp_scores)
    else:
        pred = model.predict(vec)[0]
        n_classes = len(label_enc.classes_) if label_enc else 10
        proba = np.zeros(n_classes)
        proba[pred] = 1.0

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "label": label_enc.inverse_transform([i])[0] if label_enc else str(i),
            "confidence": round(float(proba[i]) * 100, 2),
        }
        for i in top3_idx
    ]

    best_idx = top3_idx[0]
    confidence_pct = round(float(proba[best_idx]) * 100, 2)
    label_name = label_enc.inverse_transform([best_idx])[0] if label_enc else "Unknown"

    parts = label_name.split(" - ", maxsplit=1)
    subject = parts[0].strip() if len(parts) > 0 else label_name
    jenjang = parts[1].strip() if len(parts) > 1 else "Unknown"

    algo_thresh = threshold if threshold is not None else OPTIMAL_THRESHOLD_PER_ALGO.get(algorithm, 0.50)
    accepted = (confidence_pct >= algo_thresh * 100)

    return {
        "algorithm": algorithm,
        "display_name": ALGO_DISPLAY_NAME.get(algorithm, algorithm.upper()),
        "status": "accepted" if accepted else "rejected",
        "subject": subject,
        "jenjang": jenjang,
        "confidence": confidence_pct,
        "top3": top3,
        "warning": ALGORITHM_WARNINGS.get(algorithm),
    }

def classify(text: str, algorithm: str = None, threshold: float = None) -> dict:
    """
    Fungsi utama klasifikasi.

    [FIX PENTING] Sebelumnya parameter `algorithm` diterima tapi TIDAK PERNAH
    dipakai -- hasil akhir (subject/jenjang/confidence) selalu diambil dari
    algoritma dengan confidence tertinggi di antara ke-4 model, mengabaikan
    pilihan user sepenuhnya. Ini bikin fitur "pilih algoritma sebelum search"
    tidak mungkin berfungsi walau dropdown-nya sudah dibuat di frontend.

    Sekarang: kalau `algorithm` diisi (user memilih lewat setting), hasil
    akhir (subject/jenjang/confidence/status) diambil dari algoritma itu
    secara spesifik. `comparison` tetap berisi hasil KE-4 algoritma (dihitung
    sekali, tidak mahal), supaya fitur banding di history tetap dapat semua
    datanya dalam satu request -- tapi field `is_selected` menandai algoritma
    mana yang benar-benar dipakai user, terpisah dari `is_best` (algoritma
    dgn confidence tertinggi, murni informatif).
    """
    if not text or not text.strip():
        return None

    cleaned = preservation_clean(text[:1500])
    if not cleaned:
        return None

    comparison_list = []
    results_by_algo = {}
    labels_set = set()

    for algo_key in ["knn", "svm", "logreg", "rf"]:
        try:
            if get_model(algo_key):
                res = _predict_one(cleaned, algo_key, threshold)
                results_by_algo[algo_key] = res
                comparison_list.append({
                    "algorithm_key": algo_key,
                    "model": ALGO_DISPLAY_NAME.get(algo_key, algo_key.upper()),
                    "confidence": int(round(res["confidence"])),
                    "label": f"{res['subject']} - {res['jenjang']}",
                    "status": res["status"],
                    "passed": res["status"] == "accepted",
                    "subject": res["subject"],
                    "jenjang": res["jenjang"],
                    "warning": res.get("warning"),
                })
                if res["status"] == "accepted":
                    labels_set.add(f"{res['subject']} - {res['jenjang']}")
        except Exception as e:
            print(f"[CLASSIFY MODEL NOTICE] Algo '{algo_key}' skipped: {e}")

    if not comparison_list:
        return {
            "subject": "Umum",
            "jenjang": "Umum",
            "confidence": 0.50,
            "algorithm": algorithm or DEFAULT_ALGORITHM,
            "best_model": "KNN",
            "status": "accepted",
            "is_consensus": False,
            "threshold_passed": True,
            "comparison": []
        }

    # Urutan berdasar confidence tertinggi -- dipakai HANYA utk menandai
    # is_best (informasi tambahan), bukan lagi utk menentukan hasil akhir.
    comparison_sorted = sorted(comparison_list, key=lambda x: x["confidence"], reverse=True)
    top_model_key = comparison_sorted[0]["algorithm_key"]

    # [FIX] Pilih hasil akhir berdasarkan algoritma yang DIMINTA user.
    # Kalau algoritma yang diminta gagal/tidak tersedia, fallback ke default.
    chosen_key = algorithm if algorithm in results_by_algo else DEFAULT_ALGORITHM
    if chosen_key not in results_by_algo:
        chosen_key = comparison_sorted[0]["algorithm_key"]  # fallback terakhir
    chosen_result = results_by_algo[chosen_key]

    for c in comparison_list:
        c["is_best"] = (c["algorithm_key"] == top_model_key)
        c["is_selected"] = (c["algorithm_key"] == chosen_key)  # [BARU] algoritma yg dipakai user

    is_consensus = (len(labels_set) == 1) and len(comparison_list) > 0

    return {
        "subject": chosen_result["subject"],
        "jenjang": chosen_result["jenjang"],
        "confidence": round(chosen_result["confidence"] / 100.0, 4),
        "algorithm": chosen_key,
        "best_model": ALGO_DISPLAY_NAME.get(chosen_key, chosen_key.upper()),
        "status": chosen_result["status"],
        "warning": chosen_result.get("warning"),
        "is_consensus": is_consensus,
        "threshold_passed": True,
        "comparison": comparison_list,
    }

def compare_classify(text: str, threshold: float = None) -> dict:
    if not text or not text.strip():
        return None
    cleaned = preservation_clean(text[:1500])
    if not cleaned:
        return None

    results = {}
    for algo in ["knn", "svm", "logreg", "rf"]:
        try:
            if get_model(algo):
                results[algo] = _predict_one(cleaned, algo, threshold)
        except Exception as e:
            print(f"[COMPARE MODEL NOTICE] Algo '{algo}' skipped: {e}")

    return {
        "query": text,
        "results": results,
    }