from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32
)
svd = TruncatedSVD(n_components=256)

def tfidf(data) -> np.array:
    return normalize(vectorizer.fit_transform(data).astype(np.float32))

def svd_tfidf(data) -> np.array:
    return normalize(
        svd.fit_transform(vectorizer.fit_transform(data)).astype(np.float32)
    )
