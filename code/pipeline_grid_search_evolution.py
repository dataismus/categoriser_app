from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, f1_score
import nested_x_val
random_state=101


NB_alpha_range=[0.01,0.1,1.,10.]
SVM_alpha_range=[0.00001,0.0001,0.001,0.01]
max_df_range=[50,100]
min_df_range=[2, 3]

####################

param_grid= {"alpha":NB_alpha_range}
best=nested_x_val.nested_x_val_grid_search(MultinomialNB(), X, y, param_grid, show_me_params= True, score_metric=score_metric)

param_grid= {"alpha":SVM_alpha_range}
best=nested_x_val.nested_x_val_grid_search(SGDClassifier(random_state=101), X, y, param_grid, show_me_params= True, score_metric=score_metric)

####################

classifier_pipe= Pipeline([("classifier",MultinomialNB())])
param_grid= [{"classifier":[MultinomialNB()], "classifier__alpha": NB_alpha_range},
             {"classifier":[SGDClassifier(random_state=101)], "classifier__alpha": SVM_alpha_range}]

best_classifier = nested_x_val.nested_x_val_grid_search(classifier_pipe, X, y, param_grid, show_me_params= False, score_metric=score_metric)

####################

e2e_pipe= Pipeline([('vectorizer',CountVectorizer()),('classifier', MultinomialNB())])
pipe_grid= [{"classifier":[MultinomialNB()], "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "classifier__alpha": NB_alpha_range, "vectorizer__max_df":max_df_range,"vectorizer__min_df":min_df_range, "vectorizer__stop_words":[None, stopwords]},
            {"classifier":[SGDClassifier(random_state=101)], "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "classifier__alpha": SVM_alpha_range,"vectorizer__max_df":max_df_range,"vectorizer__min_df":min_df_range, "vectorizer__stop_words":[None, stopwords]}]
best_e2e=nested_x_val.nested_x_val_grid_search(e2e_pipe, raw.descr , y, pipe_grid, score_metric=score_metric)