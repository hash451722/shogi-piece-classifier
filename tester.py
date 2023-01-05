import base64
import json
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import cv2
import sklearn


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC





def read_img():
    path_current_dir = pathlib.Path(__file__).parent
    path_img = path_current_dir.joinpath("train_images", "hi", "1_10.png")

    with open(path_img, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    print(img_base64)




def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, labels:np.ndarray, path_file:pathlib.Path=None) -> None:
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")
    cm_disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
    cm_disp.plot(cmap=plt.cm.Blues)
    if path_file is None:
        path_file = pathlib.Path(__file__).parent.joinpath("cm.png")
    plt.savefig(path_file)



if __name__ == '__main__':
    read_img()


    # data = load_breast_cancer()
    # X, y = data.data, data.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # clf = SVC(random_state=0)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # plot_confusion_matrix(y_test, y_pred, data.target_names)


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html