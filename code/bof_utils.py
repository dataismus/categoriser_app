from textblob_de import TextBlobDE as TextBlobDe
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def de_blob_lemmatizer(doc):
    ws=TextBlobDe(doc).words
    return ws.lemmatize()

def lemmatizer(doc):   # preferred syntax, only this variant is affected by how the spacy_tokenizer is initialized
    tokens = nlp.tokenizer(doc)
    return [token.lemma_ for token in tokens]

def de_tokenizer_exp(doc): 
    tokenizer_exp = lambda string: nlp.tokenizer.tokens_from_list(re.compile(CountVectorizer().token_pattern).findall(string)) 
    # spacy lemmatizer combined with sklearn vectorizer's regex
    doc_spacy = tokenizer_exp(doc)
    return [token.lemma_ for token in doc_spacy]

def BoF(data,custom_tokenizer=None, tfidf=False, tokenize=True ,strip_accents=None, tokenizer=None, stop_words=None, 
        token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None):
    if tokenize:
        if custom_tokenizer:
            bag = CountVectorizer(tokenizer=custom_tokenizer, min_df=min_df,max_df=max_df, max_features=max_features,
                                ngram_range=ngram_range, stop_words=stop_words).fit(data)
        elif tfidf:
            bag = TfidfVectorizer(min_df=min_df,max_df=max_df, max_features=max_features,
                                ngram_range=ngram_range, stop_words=stop_words).fit(data)
        else:
            bag = CountVectorizer(min_df=min_df,max_df=max_df, max_features=max_features,
                                ngram_range=ngram_range, stop_words=stop_words).fit(data)
        X=bag.transform(data)
        print("Vocabulary size: {}, training set size: {} samples * {} features".format(len(bag.get_feature_names()),X.shape[0],X.shape[1]))
        print('# of tokens automatically excluded from the vocabulary:', len(bag.stop_words_))
        stopwords_eff=bag.get_stop_words()
        if stopwords_eff: print('# of stopwords that were effectively excluded :', len(stopwords_eff))
        return X, bag
    
    
# fix positional argument "data" to use in below pipeline! hint: class must inherit from sklearn basic estimator     
"""
pipe= Pipeline([('vectorizer',bof_utils.BoF()),('classifier', MultinomialNB())])
pipe_grid= [
            {"classifier":[MultinomialNB(stop_words=stopwords)], "vectorizer": [bof_utils.BoF()],
            "classifier__alpha": [0.01,0.1,1.,10.], "vectorizer__tfidf":[True, False],
             "vectorizer__max_df":[50, 100],"vectorizer__min_df":[None, 2], "vectorizer__stop_words":[None, stopwords]}]
best=nested_x_val.nested_x_val_grid_search(pipe, raw.descr , y, pipe_grid, score_metric=score_metric)
"""