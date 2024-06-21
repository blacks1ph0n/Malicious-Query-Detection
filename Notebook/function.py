# Benchmarking classifiers
import re
from pandas import read_csv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

reserved_words_df = read_csv('SQL Reserved Words.csv', dtype=str)
reserved_words = reserved_words_df['Word'].tolist()

def replace_num_hex(query):
    num_reg = re.compile(r'[0-9]+')
    hex_reg = re.compile(r'0x[A-Fa-f0-9]+\b') # regex to match hexadecimal numbers
    q = num_reg.sub('<num>', query)
    q = hex_reg.sub('<hex>', q)
    return q

# finding top 20 tokens

def find_top(is_sqlia, N):
    
    '''
    This function finds the top N tokens which are most frequent in given query type
    Parameter is_sqlia: (bool) True if we want the top N tokens of SQLIA queries
    Returns the top N tokens as a dataframe, with their respective count values
    '''
    df = token_count_df[mod_data['SQLIA']==is_sqlia]
    return df.sum(axis=0).sort_values(ascending=False).head(N)


def benchmark(clf, x, y, X_test_vec, y_test, custom_name=False):
    X_train_vec = x
    y_train = y
    from time import time
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train_vec, y_train)
    train_time = time() - t0
    print(f"train time: {train_time:.3}s")

    t0 = time()
    pred = clf.predict(X_test_vec)
    test_time = time() - t0
    print(f"test time:  {test_time:.3}s")

    score = accuracy_score(y_test, pred)
    print(f"Accuracy Score of {clf} :   {score:.3}")
    print(f"F1 score of {clf} : {f1_score(y_test, pred): .3}")
    
    confusion = confusion_matrix(y_test, pred)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sensitivity = TP / float(FN + TP)
    print(f"sensitivity of {clf} = : {sensitivity: .3}")

    specificity = TN / (TN + FP)
    print(f"specificity of {clf} = : {specificity: .3}")

    Precision = TP / float(TP + FP)
    
    print (f"Precision of {clf}= : {Precision: .3}")

    if hasattr(clf, "coef_"):
        print(f"dimensionality of {clf} : {clf.coef_.shape[1]}")
        print(f"density of {clf} : {density(clf.coef_): .3}")
        print()

    print()
    if custom_name:
        clf_descr = str(custom_name)
    else:
        clf_descr = clf.__class__.__name__
    return clf_descr, score, train_time, test_time



def eval_model(model, train_data, train_label, model_name = 'Model', cv=3):
  print(f'Evaluating {model_name}...')
  global X_test_vec
  global y_test
  train_acc_score = model.score(train_data, train_label)
  test_acc_score = model.score(X_test_vec, y_test)
  
  print('Training Accuracy = ',round(train_acc_score*100, 4), ' %')
  print('Testing Accuracy = ',round(test_acc_score*100, 4), ' %')
  
  y_predictions = model.predict(X_test_tfidfvec)
  print(f"F1 Score for {model_name}: {round(f1_score(y_test, y_predictions)*100,4)} %")
  confusion_mat = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predictions))

  cross_val_f1 = cross_val_score(model, train_data, train_label, cv=cv, scoring='f1')
  print(f'cross_validation f1-scores for {model_name} \n {cross_val_f1}')

  fpr, tpr, thresholds = roc_curve(y_test, y_predictions)
  roc_auc = auc(fpr, tpr)
  roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
  print("roc_auc = ", round(roc_auc*100,4))

  figure, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,6))
  confusion_mat.plot(ax=ax1)
  ax1.set_title('Confustion matrix')
  roc_display.plot(ax=ax2)
  ax2.set_title('ROC-AUC')
  figure.suptitle(model_name)


def func(q):
    
    '''
    This function tells us if the input is a SQL reserved keyword
    Parameter q: (str) input word
    Returns bool value which is true if q is a reserved keyword
    '''
    return q in reserved_words

def contains_reserved_words(query):
    
    '''
    This function returns true if a query contains SQL reserved keywords.
    Parameter query: (str) Input query
    Returns True if query contains reserved keywords
    '''
    words = query.split()
    return any(list(map(func, words)))
