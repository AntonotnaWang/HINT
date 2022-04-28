import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
import xgboost as xgb

def prediction(clf, feature_map_temp, is_predict_proba = False):
    if not is_predict_proba:
        concept_map = clf.predict(feature_map_temp.reshape(feature_map_temp.shape[0],
                                                           feature_map_temp.shape[1]*feature_map_temp.shape[2]).transpose(1,0))
        concept_map = concept_map.reshape(feature_map_temp.shape[1],
                                          feature_map_temp.shape[2])
    else:
        concept_map = clf.predict_proba(feature_map_temp.reshape(feature_map_temp.shape[0],
                                                                 feature_map_temp.shape[1]*feature_map_temp.shape[2]).transpose(1,0))
        concept_map = concept_map.reshape(feature_map_temp.shape[1],
                                          feature_map_temp.shape[2],
                                          concept_map.shape[1])
    return concept_map

def batch(iterable_X, iterable_y, batch_size = 1):
    l = len(iterable_X)
    for idx in range(0, l, batch_size):
        yield iterable_X[idx:min(idx + batch_size, l)], iterable_y[idx:min(idx + batch_size, l)]

def get_linear_classifier(X,y, classifier="SGD",need_recall_score=False, need_min_batch_training = False, \
    batch_num = 10, min_batch_training_round = 5):
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, shuffle= True)

    print('X train size: ',X_tr.shape)
    print('y train size: ',y_tr.shape)
    print('X valid size: ',X_vl.shape)
    print('y valid size: ',y_vl.shape)
    
    if classifier == "SGD":
        #clf = linear_model.SGDClassifier(alpha=.01, max_iter = 10000, loss = 'log')
        clf = linear_model.SGDClassifier(alpha=.01, max_iter = 10000, loss = 'modified_huber')
    elif classifier == "LogisticRegression":
        clf = linear_model.LogisticRegression(max_iter = 10000)
    
    if not need_min_batch_training:
        clf.fit(X_tr, y_tr)
    else:
        ROUNDS = min_batch_training_round
        for ROUND in range(ROUNDS):
            row_num = np.arange(X_tr.shape[0])
            np.random.shuffle(row_num)
            X_shuffle = X_tr[row_num]
            y_shuffle = y_tr[row_num]
            batcherator = batch(X_shuffle, y_shuffle, int(X_tr.shape[0] / batch_num))
            for index, (chunk_X, chunk_y) in enumerate(batcherator):
                clf.partial_fit(chunk_X, chunk_y, classes=np.unique(y))
    
    Training_score = clf.score(X_tr, y_tr)
    Validation_score = clf.score(X_vl, y_vl)
    print('Training score: ',Training_score)
    print('Validation score: ',Validation_score)
    coef = clf.coef_
    
    if len(np.unique(y_vl))==2 and need_recall_score:
        output_vl = clf.predict(X_vl)
        Recall_score = recall_score(y_vl, output_vl)
        print("recall score", Recall_score)
        return coef, clf, Validation_score, Recall_score
    else:
        return coef, clf, Validation_score

def get_xgb_classifier(X,y,model_para = False):
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, shuffle= True)
    dtrain = xgb.DMatrix(data=X_tr,label=y_tr)
    dtest = xgb.DMatrix(data=X_vl,label=y_vl)

    param = {"tree_method": "gpu_hist", 'max_depth':10, 'n_jobs':4, 'eta':1, 'objective':'binary:logistic' }
    num_round = 50
    
    if model_para:
        bst = xgb.train(param, dtrain, num_round, xgb_model='model_para.model')
    else:
        bst = xgb.train(param, dtrain, num_round)
    bst.save_model('model_para.model')
    bst.set_param({"predictor": "gpu_predictor"})
    
    # make prediction
    preds_dtest = bst.predict(dtest)
    preds_dtrain = bst.predict(dtrain)

    # evaluate predictions
    train_accuracy = accuracy_score(y_tr, [round(value) for value in preds_dtrain])
    print("Training Accuracy: %.2f%%" % (train_accuracy * 100.0))
    val_accuracy = accuracy_score(y_vl, [round(value) for value in preds_dtest])
    print("Validation Accuracy: %.2f%%" % (val_accuracy * 100.0))
    
    return bst

def get_sampled_DMatrix(X, y):
    chosen_locs = np.concatenate((np.random.choice(np.where(y==0)[0], size=1400, replace=False),
                                  np.random.choice(np.where(y==1)[0], size=700, replace=False)))
    return xgb.DMatrix(data=X[chosen_locs, :],label=y[chosen_locs]), X[chosen_locs, :]