import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report as report


def preprocessing(results, truth):
    # preprocessing
    results.loc[truth['before'] == truth['after'], 'truth'] = 'RemainSelf'
    results.loc[truth['before'] != truth['after'], 'truth'] = 'ToBeNormalized'
    truth['class'] = ''
    truth.loc[truth['before'] != truth['after'], 'class'] = 'ToBeNormalized'
    truth.loc[truth['before'] == truth['after'], 'class'] = 'RemainSelf'
    return results, truth


def flatten(li):
    return [item for sublist in li for item in sublist]


def f1_scores(results, truth):
    print(report(flatten(truth), flatten(results)))


def confusion_matrix(results, truth, classes):
    matrix = cm(flatten(truth), flatten(results))
    plot_confusion_matrix(matrix, classes=classes,
                          title='Confusion Matrix')


def pr_curve(results, truth, lang):
    truth.loc[truth['class'] == 'ToBeNormalized', 'class'] = 1
    truth.loc[truth['class'] == 'RemainSelf', 'class'] = 0
    results.loc[results['class'] == 'ToBeNormalized', 'class'] = 1
    results.loc[results['class'] == 'RemainSelf', 'class'] = 0

    average_precision = average_precision_score(truth['class'].tolist(), results['class'].tolist())
    precision, recall, threshold = precision_recall_curve(truth['class'].tolist(), results['class'].tolist())

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AP={0:0.2f} [{1}]'.format(average_precision, lang))
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
