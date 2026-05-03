import re
import spacy

nlp = spacy.load("en_core_web_sm")

# ---------------- HINGLISH MAP ----------------
HINGLISH_MAP = {
    "hai kya": "available",
    "milega": "available",
    "milta": "available",
    "padha": "available",
    "kitna": "stock",
    "kitni": "stock",
    "kitne": "stock",
    "dikhao": "show",
    "dikhana": "show",
    "sab": "all",
    "maal": "product",
    "saman": "product",
    "namaste": "hello",
    "shukriya": "thanks",
    "alvida": "bye"
}

# ---------------- INTENTS ----------------
INTENTS = {
    "greet": ["hi","hello","hey","namaste","ram"],
    "bye": ["bye","exit","quit","alvida"],
    "thanks": ["thanks","thank","shukriya"],
    "show": ["show","list","display","all"],
    "stock": ["stock","quantity","kitna","available"]
}

STOP_WORDS = [
    "is","are","am","the","a","an","please","pls","plz",
    "product","item","items","ka","ki","ke","ko","hai",
    "kya","hai","haii","haiii"
]

# ---------------- NORMALIZE TEXT ----------------
def normalize_text(text: str):
    text = text.lower()

    # hinglish replace FIRST
    for k, v in HINGLISH_MAP.items():
        text = text.replace(k, v)

    # remove special chars
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

    # remove repeated letters (haaiii → hai)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    return text.strip()


# ---------------- NLP ANALYSIS ----------------
def analyze_text(text: str):
    text = normalize_text(text)

    doc = nlp(text)

    tokens = []
    for token in doc:
        word = token.lemma_.lower()

        if word not in STOP_WORDS and word.isalpha():
            tokens.append(word)

    return {
        "tokens": tokens,
        "clean_text": text
    }


# ---------------- INTENT DETECTION ----------------
def detect_intent(tokens):
    # priority order
    order = ["greet", "bye", "thanks", "show", "stock"]

    for intent in order:
        for t in tokens:
            if t in INTENTS[intent]:
                return intent

    return "search"
