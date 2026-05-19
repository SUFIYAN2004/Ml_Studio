"""
text_cleaner.py
Pure-Python / regex text cleaning utilities.
NLTK is used ONLY if already available on the system — never required.
All operations degrade gracefully.
"""
import re
import string
from typing import List


# ── Optional NLTK ──────────────────────────────────────────────────────────────
try:
    from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords
    import nltk
    _NLTK_OK = True
    _STEMMER_PORTER = PorterStemmer()
    _STEMMER_LANCASTER = LancasterStemmer()
    _STEMMER_SNOWBALL = SnowballStemmer("english")
    _LEMMATIZER = WordNetLemmatizer()
    try:
        _STOP_WORDS = set(nltk_stopwords.words("english"))
    except Exception:
        _STOP_WORDS = set()
except Exception:
    _NLTK_OK = False
    _STOP_WORDS = set()

# Fallback English stopwords (top-200) if NLTK unavailable
_FALLBACK_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn","just","also","even","back","still","way","take","made",
    "make","get","go","first","last","many","new","much","one","two","would",
    "could","may","might","shall","us","however","although","though","since",
    "whether","while","unless","until","rather","quite","almost","already",
}

if not _STOP_WORDS:
    _STOP_WORDS = _FALLBACK_STOPWORDS

# ── Contractions map ───────────────────────────────────────────────────────────
CONTRACTIONS = {
    "can't":"cannot","won't":"will not","n't":" not","i'm":"i am",
    "you're":"you are","he's":"he is","she's":"she is","it's":"it is",
    "we're":"we are","they're":"they are","i've":"i have","you've":"you have",
    "we've":"we have","they've":"they have","i'd":"i would","you'd":"you would",
    "he'd":"he would","she'd":"she would","we'd":"we would","they'd":"they would",
    "i'll":"i will","you'll":"you will","he'll":"he will","she'll":"she will",
    "we'll":"we will","they'll":"they will","that's":"that is","what's":"what is",
    "there's":"there is","here's":"here is","let's":"let us","who's":"who is",
    "isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not",
    "don't":"do not","doesn't":"does not","didn't":"did not","haven't":"have not",
    "hasn't":"has not","hadn't":"had not","couldn't":"could not","wouldn't":"would not",
    "shouldn't":"should not","mightn't":"might not","mustn't":"must not",
}


# ── Individual cleaning functions ──────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    return text.lower()


def expand_contractions(text: str) -> str:
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS) + r')\b', re.IGNORECASE)
    def replace(m):
        return CONTRACTIONS.get(m.group().lower(), m.group())
    return pattern.sub(replace, text)


def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_emails(text: str) -> str:
    return re.sub(r'\S+@\S+\.\S+', ' ', text)


def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return re.sub(r'@\w+', ' ', text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtag tokens."""
    return re.sub(r'#\w+', ' ', text)


def remove_digits(text: str) -> str:
    return re.sub(r'\d+', ' ', text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))


def remove_special_characters(text: str) -> str:
    """Remove non-alphanumeric, non-whitespace characters."""
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(' ', text)


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return ' '.join(w for w in tokens if w.lower() not in _STOP_WORDS)


def remove_short_words(text: str, min_len: int = 2) -> str:
    tokens = text.split()
    return ' '.join(w for w in tokens if len(w) >= min_len)


def apply_stemming_porter(text: str) -> str:
    if _NLTK_OK:
        try:
            return ' '.join(_STEMMER_PORTER.stem(w) for w in text.split())
        except Exception:
            pass
    # Simple regex-based fallback
    return re.sub(r'(ing|ed|er|est|ly|tion|ness|ment|ful|ous|ive|ize|ise)$', '', text)


def apply_stemming_lancaster(text: str) -> str:
    if _NLTK_OK:
        try:
            return ' '.join(_STEMMER_LANCASTER.stem(w) for w in text.split())
        except Exception:
            pass
    return apply_stemming_porter(text)


def apply_stemming_snowball(text: str) -> str:
    if _NLTK_OK:
        try:
            return ' '.join(_STEMMER_SNOWBALL.stem(w) for w in text.split())
        except Exception:
            pass
    return apply_stemming_porter(text)


def apply_lemmatization(text: str) -> str:
    if _NLTK_OK:
        try:
            return ' '.join(_LEMMATIZER.lemmatize(w) for w in text.split())
        except Exception:
            pass
    # Lightweight suffix-stripping fallback
    rules = [('ies', 'y'), ('ied', 'y'), ('ves', 'f'), ('oes', 'o'),
             ('ses', 's'), ('xes', 'x'), ('zes', 'z'), ('hes', 'h'),
             ('ing', ''), ('tion', 'te'), ('ness', ''), ('ment', '')]
    def lemma(word):
        for suffix, repl in rules:
            if word.endswith(suffix) and len(word) - len(suffix) > 2:
                return word[:-len(suffix)] + repl
        return word
    return ' '.join(lemma(w) for w in text.split())


def apply_custom_regex(text: str, pattern: str, replacement: str = ' ') -> str:
    """Apply a user-defined regex substitution."""
    try:
        return re.sub(pattern, replacement, text)
    except re.error:
        return text


# ── Master pipeline ────────────────────────────────────────────────────────────

STEP_REGISTRY = {
    "lowercase":            ("Lowercase", to_lowercase, "Convert all text to lowercase."),
    "expand_contractions":  ("Expand contractions", expand_contractions, "can't → cannot, won't → will not, etc."),
    "remove_html":          ("Remove HTML tags", remove_html_tags, "Strip <b>, <p>, <a href=...>, etc."),
    "remove_urls":          ("Remove URLs", remove_urls, "Remove http://, https://, www."),
    "remove_emails":        ("Remove email addresses", remove_emails, "Remove user@domain.com."),
    "remove_mentions":      ("Remove @mentions", remove_mentions, "Remove Twitter/social @username."),
    "remove_hashtags":      ("Remove #hashtags", remove_hashtags, "Remove #topic hashtags."),
    "remove_emojis":        ("Remove emojis", remove_emojis, "Strip unicode emoji characters."),
    "remove_digits":        ("Remove digits", remove_digits, "Remove 0–9 numerals."),
    "remove_punctuation":   ("Remove punctuation", remove_punctuation, "Remove .,!?;:'\"-()[] etc."),
    "remove_special_chars": ("Remove special characters", remove_special_characters, "Keep only alphanumeric + whitespace."),
    "remove_stopwords":     ("Remove stopwords", remove_stopwords, "Remove common English stopwords (the, is, at, ...)."),
    "remove_short_words":   ("Remove short words (< 3 chars)", lambda t: remove_short_words(t, 3), "Drop tokens with fewer than 3 characters."),
    "whitespace":           ("Normalize whitespace", remove_extra_whitespace, "Collapse multiple spaces/newlines to single space."),
    "lemmatize":            ("Lemmatization", apply_lemmatization, "Reduce words to base form (running → run). Uses NLTK WordNet if available."),
    "stem_porter":          ("Stemming — Porter", apply_stemming_porter, "Aggressive suffix stripping (Porter algorithm)."),
    "stem_lancaster":       ("Stemming — Lancaster", apply_stemming_lancaster, "More aggressive than Porter."),
    "stem_snowball":        ("Stemming — Snowball", apply_stemming_snowball, "Language-aware stemmer (recommended over Porter)."),
}

# Recommended ordering (applied in this sequence regardless of checkbox order)
STEP_ORDER = [
    "lowercase", "expand_contractions", "remove_html", "remove_urls",
    "remove_emails", "remove_mentions", "remove_hashtags", "remove_emojis",
    "remove_digits", "remove_punctuation", "remove_special_chars",
    "remove_stopwords", "remove_short_words",
    "lemmatize", "stem_porter", "stem_lancaster", "stem_snowball",
    "whitespace",
]


def clean_text(text: str, steps: List[str], custom_pattern: str = "", custom_replacement: str = " ") -> str:
    """
    Apply the selected cleaning steps in the recommended order.
    steps: list of keys from STEP_REGISTRY
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    for key in STEP_ORDER:
        if key in steps:
            fn = STEP_REGISTRY[key][1]
            text = fn(text)

    if custom_pattern.strip():
        text = apply_custom_regex(text, custom_pattern.strip(), custom_replacement)

    # Always clean whitespace at the very end
    text = remove_extra_whitespace(text)
    return text


def clean_series(series, steps: List[str], custom_pattern: str = "", custom_replacement: str = " "):
    """Clean an entire pandas Series."""
    return series.fillna("").astype(str).apply(
        lambda t: clean_text(t, steps, custom_pattern, custom_replacement)
    )
