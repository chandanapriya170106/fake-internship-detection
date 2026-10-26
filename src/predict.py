"""Load the trained model and classify a posting."""
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.preprocess import clean_text, red_flag_features

_BUNDLE = None
FLAG_LABELS = {
    "asks_money": "Asks for money / fees",
    "free_email": "Uses a free email ID (gmail/yahoo...)",
    "urgency": "Urgency / pressure language",
    "instant_selection": "Instant selection, no interview",
    "whatsapp_telegram": "Contact only via WhatsApp/Telegram",
}


def _load(path="models/fake_internship_model.joblib"):
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def predict_posting(text: str) -> dict:
    b = _load()
    flags = red_flag_features(text)
    X = hstack([b["tfidf"].transform([clean_text(text)]),
                csr_matrix(np.array([[flags[c] for c in b["flag_cols"]]]))]).tocsr()
    model = b["model"]
    if hasattr(model, "predict_proba"):
        p_fake = float(model.predict_proba(X)[0][1])
    else:  # LinearSVC -> squash decision score into 0-1
        p_fake = float(1 / (1 + np.exp(-model.decision_function(X)[0])))
    triggered = [FLAG_LABELS[k] for k in FLAG_LABELS if flags.get(k)]
    if not flags["has_company_website"]:
        triggered.append("No website / verifiable link")
    return {
        "label": "Fake" if p_fake >= 0.5 else "Genuine",
        "confidence": p_fake if p_fake >= 0.5 else 1 - p_fake,
        "p_fake": p_fake,
        "red_flags": triggered,
        "model": b["name"],
    }


if __name__ == "__main__":
    demo = ("Urgent!!! Pay registration fee Rs 999 for offer letter. No interview. "
            "Contact hr123@gmail.com on WhatsApp. Earn Rs 50000 from home.")
    print(predict_posting(demo))
