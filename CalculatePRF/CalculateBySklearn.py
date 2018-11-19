from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
y_true=[2,1,0,1,2,0]
y_pred=[2,0,0,1,2,1]

C=confusion_matrix(y_true, y_pred)

precision_score(y_true, y_pred, average='macro')
precision_score(y_true, y_pred, average='micro')
recall_score(y_true, y_pred, average='macro')
recall_score(y_true, y_pred, average='micro')