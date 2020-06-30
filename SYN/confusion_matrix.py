import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play withs
df_confusion = np.array([[115, 2, 2, 3, 1, 4, 1, 0],
                        [5, 117, 2, 1, 0, 2, 1, 0],
                        [1, 4, 119, 1, 1, 0, 0, 2],
                        [2, 0, 1, 121, 3, 0, 0, 1],
                        [2, 0, 0, 1, 122, 2, 0, 1],
                        [1, 0, 0, 0, 1, 124, 1, 1],
                        [1, 0, 0, 0, 0, 1, 124, 2],
                        [4, 0, 0, 0, 0, 0, 2, 122]])

np.set_printoptions(precision=2)
df_nm = df_confusion.astype('float') / df_confusion.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,8), dpi=120)
ind_array = np.arange(8)
x, y = np.meshgrid(ind_array, ind_array)



def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.xticks(tick_marks, [-4,-3,-2,-1,0,1,2,3], rotation=45)
    plt.yticks(tick_marks, [-4,-3,-2,-1,0,1,2,3])
    #plt.tight_layout()
    plt.ylabel("Ground-Truth")
    plt.xlabel("Predicted")

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = df_nm[y_val][x_val]
        print c
        plt.text(x_val, y_val, "%0.2f" % (c,), color='green', fontsize=7, va='center', ha='center')


plot_confusion_matrix(df_nm)

plt.savefig("con_matrix.png")
plt.show()
