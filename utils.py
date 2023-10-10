import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('book')


def hashtags(caption):
    hashtag = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(caption)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    # X = len(filtered_sentence)
    for i in filtered_sentence:
        hashtag.append('#' + i)
    return hashtag
