from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
svd = TruncatedSVD(n_components=256)


def tfidf(data):
    return svd.fit_transform(vectorizer.fit_transform(data))
