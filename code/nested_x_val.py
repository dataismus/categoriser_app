####### F1 score averaging technique: macro or micro?
####### Do we have a preference for Recall *OR* Precision            ---> Micro
####### ... or care of accuracy of prediction for each class equally ---> Macro
# score_metric="accuracy"   ### UNSUITABLE for skewed class datasets
# score_metric="f1_micro"
# score_metric="f1_macro"

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np


def nested_x_val_grid_search(estimator, X,y, param_grid, show_me_params= False,n_splits=3, n_jobs=-1, score_metric="f1_macro", random_state=101):
    kfold=StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=random_state)
    param_grid= param_grid
    grid_search_NB=GridSearchCV(estimator, param_grid, cv=kfold, n_jobs=n_jobs, refit=score_metric)

    nested_grid_search=cross_validate(grid_search_NB, X,y, cv=kfold, scoring=score_metric, return_train_score=True, return_estimator=True)
    # print('Test set scores of each split: '+(len(nested_grid_search['test_score'])*'{:.3f} ').format(*nested_grid_search['test_score']))
    print('Average '+score_metric+' on the test set: {:.3f}'.format(np.mean(nested_grid_search['test_score'])))
    print('Average '+score_metric+' score on the train set: {:.3f}'.format(np.mean(nested_grid_search['train_score'])))

    if show_me_params: print('\nOptimal hyperparameter acquired by nested grid search: ', nested_grid_search["estimator"][0].best_estimator_.get_params())
    return nested_grid_search["estimator"][0].best_estimator_


def model_mvp(models, X, y , n_splits=3, n_jobs=-1, score_metric="f1_macro", random_state=101):
    estimators={}
    kfold=StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=random_state)
    for model in models:

        cv=cross_validate(model, X,y, cv=kfold, scoring=score_metric, return_train_score=True, return_estimator=True)
        train_score=np.mean(cv['train_score'])
        test_score =np.mean(cv['test_score'])

        print('{:>56}'.format('train vs. test'))
        print(score_metric+' of default {:>20}: {:>5.2f}     {:>4.2f}'.format(type(model).__name__, train_score, test_score))
        estimators[type(model).__name__]=cv["estimator"][0]
    return estimators