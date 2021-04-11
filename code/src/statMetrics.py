
import pdb
import numpy as np

def get_confusion_matrix_metrics(confusion_matrix):
    tp = confusion_matrix[0,0]
    tn = confusion_matrix[1,1]
    fp = confusion_matrix[0,1]
    fn = confusion_matrix[1,0]
    if fp+fp==0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if tp+fn==0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    if precision+recall==0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

def get_multiclass_confusion_matrix_metrics(confusion_matrix):

    def getClassConfusionMetrics(class_index, confusion_matrix):
        tp = confusion_matrix[i,i]
        tn = np.delete(np.delete(confusion_matrix, i, axis=0), i, axis=1).sum()
        fp = np.delete(confusion_matrix[i,:], i).sum()
        fn = np.delete(confusion_matrix[:,i], i).sum()
        if fp+fp==0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        if tp+fn==0:
            recall = 0
        else:
            recall = tp/(tp+fn)
        if precision+recall==0:
            f1 = 0
        else:
            f1 = 2*precision*recall/(precision+recall)
        return precision, recall, f1

    nClasses = confusion_matrix.shape[0]
    class_metrics = np.empty((nClasses, 3))
    for i in range(nClasses):
        class_metrics[i,:] = getClassConfusionMetrics(class_index=i, confusion_matrix=confusion_matrix)
    macro_metrics = np.mean(class_metrics, axis=0)

    return class_metrics, macro_metrics

