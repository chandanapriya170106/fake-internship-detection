"""Text cleaning + hand-crafted red-flag features."""
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FREE_EMAIL = re.compile(r"@(gmail|yahoo|outlook|hotmail|rediffmail)\.", re.I)
FEE_WORDS = re.compile(
    r"(registration fee|security deposit|training fee|kit fee|pay (a )?fee|"
    r"processing fee|refundable|deposit|pay rs|pay inr|send money|payment required)", re.I)
URGENCY = re.compile(r"(urgent|apply immediately|limited seats|last date today|hurry|act now|within 24 hours)", re.I)
INSTANT = re.compile(r"(no interview|direct selection|instant (selection|offer)|selected without|100% selection|guaranteed (job|offer|selection))", re.I)
MESSAGING = re.compile(r"(whatsapp|telegram)", re.I)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/punctuation/digits, remove stop words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def red_flag_features(raw: str) -> dict:
    """Signals from the 'Red Flags to Detect' slide."""
    raw = str(raw)
    return {
        "asks_money": int(bool(FEE_WORDS.search(raw))),
        "free_email": int(bool(FREE_EMAIL.search(raw))),
        "urgency": int(bool(URGENCY.search(raw))),
        "instant_selection": int(bool(INSTANT.search(raw))),
        "whatsapp_telegram": int(bool(MESSAGING.search(raw))),
        "has_company_website": int(bool(re.search(r"(https?://|www\.)", raw))),
        "exclamations": min(raw.count("!"), 10),
        "text_length": min(len(raw) / 1000.0, 5.0),
    }


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]


def build_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Merge text columns, then add `clean` text and red-flag feature columns."""
    df = df.copy()
    for c in TEXT_COLS:
        if c not in df.columns:
            df[c] = ""
    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df["raw"] = df[TEXT_COLS].agg(" ".join, axis=1)
    df["clean"] = df["raw"].apply(clean_text)
    flags = pd.DataFrame([red_flag_features(r) for r in df["raw"]], index=df.index)
    return pd.concat([df, flags], axis=1)
