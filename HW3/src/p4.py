'''
Created on Nov 24, 2017
 
@author: Amir
'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from p1 import getInput

import warnings
warnings.filterwarnings("ignore")

# Import some data to play with
X, _, y = getInput()

pca = PCA(30)
x = pca.fit_transform(X.T)
yStrain = y[:,1]
yMedium = y[:,2]
yEnv_Pert = y[:,3]
yGene_Pert = y[:,4]


classifierStrain = OneVsRestClassifier(svm.SVC(C= 5))
classifierMedium = OneVsRestClassifier(svm.SVC(C= 5))
classifierEnv_Pert = OneVsRestClassifier(svm.SVC(C= 5))
classifierGene_Pert = OneVsRestClassifier(svm.SVC(C= 5))

strainClass=['BW25113', 'CG2', 'DH5alpha', 'MG1655','P2', 'P4X','W3110','rpoA14','rpoA27','rpoD3']
mediumClass=["MD001", "MD002", "MD003", "MD004", "MD005", "MD006", "MD007", "MD008", "MD009", "MD010",
              "MD011", "MD012", "MD013", "MD014", "MD015", "MD016", "MD017", "MD018"]
envClass= ["Indole", "O2-starvation", "RP-overexpress", "Antibacterial", "Carbon-limitation",
            "Dna-damage", "Zinc-limitation", "none"]
geneClass= ["appY_KO","arcA_KO","argR_KO", "cya_KO", "fis_OE", "fnr_KO", "frdC_KO", "na_WT", "oxyR_KO",
             "rpoS_KO", "soxS_KO", "tnaA_KO"]

x = pca.components_.T
# Binarize the output
yStrain = label_binarize(yStrain, classes=strainClass)
yMed = label_binarize(yMedium, classes=mediumClass)
yEnv = label_binarize(yEnv_Pert, classes=envClass)
yGene = label_binarize(yGene_Pert, classes=geneClass)

numClassesStrain, numClassesMed, numClassesEnv, numClassesGene = yStrain.shape[1],yMed.shape[1],yEnv.shape[1], yGene.shape[1]


# kf = KFold(n_splits=10, shuffle = True, random_state = 69)
# for train_index, test_index in kf.split(x):
#     XStrain_train, XStrain_test, yStrain_train, yStrain_test = x[train_index], x[test_index],yStrain[train_index], ystrain[test_index]
#     xMed_train, xMed_test, yMed_train, yMed_test= x[train_index], x[test_index],yMed[train_index], yMed[test_index]
#     XEnv_train, XEnv_test, yEnv_train, yEnv_testx = x[train_index], x[test_index],yEnv[train_index], yEnv[test_index]
#     XGene_train, XGene_test, yGene_train, yGene_testx = x[train_index], x[test_index],yGene[train_index], yGene[test_index]
print(x.shape, yStrain.shape)
XStrain_train, XStrain_test, yStrain_train, yStrain_test = train_test_split(x, yStrain, test_size=.2, random_state=0)
xMed_train, xMed_test, yMed_train, yMed_test = train_test_split(x, yMed, test_size=.2, random_state=0)
xEnv_train, xEnv_test, yEnv_train, yEnv_test = train_test_split(x, yEnv, test_size=.2, random_state=0)
xGene_train, xGene_test, yGene_train, yGene_test = train_test_split(x, yGene, test_size=.2, random_state=0)

yStrain_score = classifierStrain.fit(XStrain_train, yStrain_train).decision_function(XStrain_test)
yMed_score = classifierMedium.fit(xMed_train, yMed_train).decision_function(xMed_test)
yEnv_score = classifierEnv_Pert.fit(xEnv_train, yEnv_train).decision_function(xEnv_test)
yGene_score = classifierGene_Pert.fit(xGene_train, yGene_train).decision_function(xGene_test)


# Learn to predict each class against the other

# classNum = 3
# # Compute ROC curve and ROC area for each class
Sfpr = dict()
Stpr = dict()
Sroc_auc = dict()
Mfpr = dict()
Mtpr = dict()
Mroc_auc = dict()
Efpr = dict()
Etpr = dict()
Eroc_auc = dict()
Gfpr = dict()
Gtpr = dict()
Groc_auc = dict()
sPrecision, sRecall, Spr_auc = dict(), dict(), dict()
mPrecision, mRecall, Mpr_auc = dict(), dict(), dict()
ePrecision, eRecall, Epr_auc = dict(), dict(), dict()
gPrecision, gRecall, Gpr_auc = dict(), dict(), dict()

for i in range(numClassesStrain):
    Sfpr[i], Stpr[i], thresholds = roc_curve(yStrain_test[:, i], yStrain_score[:, i])
    sPrecision[i], sRecall[i], _ = precision_recall_curve(yStrain_test[:, i], yStrain_score[:, i])
    Sroc_auc[i] = auc(Sfpr[i], Stpr[i])
    Spr_auc[i] = average_precision_score(yStrain_test[:, i], yStrain_score[:, i])
for i in range(numClassesMed):
    Mfpr[i], Mtpr[i], thresholds = roc_curve(yMed_test[:, i], yMed_score[:, i])
    mPrecision[i], mRecall[i], _  = precision_recall_curve(yMed_test[:, i], yMed_score[:, i])
    Mroc_auc[i] = auc(Mfpr[i], Mtpr[i])
    Mpr_auc[i] = average_precision_score(yMed_test[:, i], yMed_score[:, i])
for i in range(numClassesEnv):
    Efpr[i], Etpr[i], thresholds = roc_curve(yEnv_test[:, i], yEnv_score[:, i])
    ePrecision[i], eRecall[i], _  = precision_recall_curve(yEnv_test[:, i], yEnv_score[:, i])
    Eroc_auc[i] = auc(Efpr[i], Etpr[i])
    Epr_auc[i] = average_precision_score(yEnv_test[:, i], yEnv_score[:, i])
for i in range(numClassesGene):
    Gfpr[i], Gtpr[i], thresholds = roc_curve(yGene_test[:, i], yGene_score[:, i])
    gPrecision[i], gRecall[i], _  = precision_recall_curve(yGene_test[:, i], yGene_score[:, i])
    Groc_auc[i] = auc(Gfpr[i], Gtpr[i])
    Gpr_auc[i] = average_precision_score(yGene_test[:, i], yGene_score[:, i])

     
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


fig = plt.figure()
fig.suptitle("ROC curves                                          PR curves")
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

       
for i, name in zip(range(4), ("Strain", "Medium", "Env_Pert", "Gene_Pert")):
    ax = fig.add_subplot(410+i+1)
    ax.set_title(name)
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

ax = []
for i in range(8):
    ax.append(fig.add_subplot(420 + i+1))
    ax[i].set_xlim([0.0, 1.0])
#     ax[i].set_prop_cycle(cycler('marker', ['.', 's', 'x', 'd']) *
#                    cycler('color', ['b', 'g', 'r', 'y']))
    ax[i].set_ylim([0.0, 1.05])
    if i%2 == 0: 
        ax[i].plot([0, 1], [0, 1], color='navy', linestyle='--')
for j, cl in zip(range(numClassesStrain), strainClass):
    ax[0].plot(Sfpr[j], Stpr[j], label= cl + ' (area = %0.2f)' % Sroc_auc[j])
    ax[1].plot(sPrecision[j], sRecall[j], label =  cl + ' (area = %0.2f)' % Spr_auc[j])

for j, cl in zip(range(numClassesMed), mediumClass):
    if i >= 9:
        ax[2].plot(Mfpr[j], Mtpr[j], label = cl[-2:] + '  (%0.2f)' % Mroc_auc[j])
        ax[3].plot(mPrecision[j], mRecall[j], label = cl[-2:] + '  (%0.2f)' % Mpr_auc[j])
    ax[2].plot(Mfpr[j], Mtpr[j], label = cl[-1:] + '  (%0.2f)' % Mroc_auc[j])
    ax[3].plot(mPrecision[j], mRecall[j], label = cl[-1:] + '  (%0.2f)' % Mpr_auc[j])


for j, cl in zip(range(numClassesEnv), envClass):
    ax[4].plot(Efpr[j], Etpr[j], label = cl + ' (area = %0.2f)' % Eroc_auc[j])
    ax[5].plot(ePrecision[j], eRecall[j], label = cl + ' (area = %0.2f)' % Epr_auc[j])

for j, cl in zip(range(numClassesGene), geneClass):
    ax[6].plot(Gfpr[j], Gtpr[j], label = cl + ' (area = %0.2f)' % Groc_auc[j])
    ax[7].plot(gPrecision[j], gRecall[j], label = cl + ' (area = %0.2f)' % Gpr_auc[j])


ax[0].legend(bbox_to_anchor=(-.08, 1), fontsize = 7)
ax[1].legend(bbox_to_anchor=(1.02, 1), fontsize = 7)
ax[2].legend(bbox_to_anchor=(-.08, 1), fontsize = 7, ncol = 2)
ax[3].legend(bbox_to_anchor=(1.02, 1), fontsize = 7, ncol = 2)
ax[4].legend(bbox_to_anchor=(-.08, 1), fontsize = 7)
ax[5].legend(bbox_to_anchor=(1.02, 1), fontsize = 7)
ax[6].legend(bbox_to_anchor=(-.08, 1), fontsize = 7)
ax[7].legend(bbox_to_anchor=(1.02, 1), fontsize = 7)
plt.show()
# print(roc_auc)
#         ax[i].plot(fpr[j], tpr[j], color='darkorange',
#                     label='ROC curve (area = %0.2f)' % roc_auc[j])