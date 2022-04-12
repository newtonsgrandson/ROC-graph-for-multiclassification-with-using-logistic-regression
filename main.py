%clear
from random import randint
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle

@ignore_warnings(category=ConvergenceWarning)
def classification():
    global X, y, model, X_train, X_test, y_train, y_test
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = predictions.round()
    predictions = pd.Series(predictions)
    predictions.index = y_test.index
    return y_test, predictions
 
@ignore_warnings(category=ConvergenceWarning)    
def reportDifference():
    global X, y, random_state_pos, report
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state_pos)
    newModel = LogisticRegression(random_state = random_state_pos)
    newModel.fit(X_train, y_train)
    predictions = newModel.predict(X_test)
    reportNew = classification_report(y_test, predictions, output_dict = True)
    report, reportNew = pd.DataFrame(report), pd.DataFrame(reportNew).transpose()
    print("The report of normal one which test_size = 0.33")
    print(report)
    print("The differences report - reportNew")
    print(report - reportNew)

@ignore_warnings(category=ConvergenceWarning)   
def ROCvalues(target):
    global X, y, random_state_pos
    newY = [1 if y.iloc[i] == target else 0 for i in range(y.__len__())]
    X_train, X_test, y_train, y_test = train_test_split(X, newY, random_state=random_state_pos)
    model1 = LogisticRegression(random_state = random_state_pos)
    model1.fit(X_train, y_train)
    proba = [i for i in list(pd.DataFrame(model1.predict_proba(X_test))[1])]
    y_test, proba = np.array(y_test), np.array(proba)
    return y_test, proba

def binarizeClassification():
    global y_test, predictions
    y_testB = [1 if int(y_test.iloc[i]) / 2 == j else 0 for i in range(y_test.__len__()) for j in range(5)]
    predictionsB = [1 if int(predictions.iloc[i]) / 2 == j else 0 for i in range(predictions.__len__()) for j in range(5)]       
    y_testB, predictionsB = pd.Series(y_testB),  pd.Series(predictionsB)
    return y_testB, predictionsB

def showClassification(randRow):
    global model
    img = randRow.values.reshape(int(math.sqrt(randRow.__len__())), int((math.sqrt(randRow.__len__()))))
    print("The predicted value was: " + str(model.predict(img.reshape(1, randRow.__len__()))[0]))
    plt.imshow(img)

#Preprocessing
random_state_pos = 42
ld = load_digits()
X = pd.DataFrame(ld.data, columns = ld.feature_names)
y = pd.Series(ld.target)
indexOdd = [i for i in range(len(y.astype(int))) if int(y[i]) % 2 == 1]
X.drop(indexOdd, axis = 0,  inplace = True)
y.drop(indexOdd, axis = 0,  inplace = True)

#Classification
model = LogisticRegression(random_state=random_state_pos)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state_pos)
y_test, predictions = classification()

#Model Performance
report = classification_report(y_test, predictions, output_dict=(True))
report = pd.DataFrame(report).transpose()

#ROC
y_testB, predictionsB = binarizeClassification()
avgTotalRocAucValue = roc_auc_score(y_testB, predictionsB)
print("Average all targets roc_auc_score with binarize technic: " + str(avgTotalRocAucValue))
n_classes = 5
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y_testP, predictionsP = ROCvalues(i * 2)
    fpr[i], tpr[i], _ = roc_curve(y_testP, predictionsP)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_testP.ravel(), predictionsP.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i * 2, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()

rand = randint(0,X_test.__len__() - 1);
randRow = X_test.iloc[rand, :]
showClassification(randRow)
reportDifference()