import base64
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, precision_score

import classify_piece_tester


class OnnxTester():
    def __init__(self) -> None:
        self.cp = classify_piece_tester.ClassifyPiece()
        self.labels = self.set_labels()

        self.test_data = self.set_test_data()
        self.test_labels, self.test_imgs = self.preprocess(self.test_data)

        path_current_dir = pathlib.Path(__file__).parent
        self.path_models_dir = path_current_dir.joinpath("models")


    def set_test_data(self):
        '''
        29種類すべてのテストデータが必要
        '''
        labels_dummy = ['bfu', 'bgi', 'bhi', 'bka','bke', 'bki', 'bky', 'bng', 'bnk', 'bny',
        'bou', 'bry', 'bto', 'bum', 'emp', 'wfu', 'wgi', 'whi', 'wka', 'wke', 'wki', 'wky',
        'wng', 'wnk', 'wny', 'wou', 'wry', 'wto', 'wum']
        imgs_dummy = np.random.randn(29, 3, 64, 64)  # (N, C, H, W)
        return (labels_dummy, imgs_dummy)


    def preprocess(self, test_data):
        test_labels = list(map(lambda x: self.cp.class_to_idx[x], test_data[0]))
        test_imgs = test_data[1] * 1.0
        return (np.array(test_labels, dtype=np.int32), test_imgs)


    def set_labels(self):
        labels = []
        for i in range(len(self.cp.idx_to_class)):
            labels.append(self.cp.idx_to_class[str(i)])
        return np.array(labels)


    def run(self):
        pred_labels = self.cp.run(self.test_imgs)
        self.plot_confusion_matrix(self.test_labels, pred_labels, self.labels)


    def plot_confusion_matrix(self, y_true:np.ndarray, y_pred:np.ndarray, labels:np.ndarray, path_file:pathlib.Path=None) -> None:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")
        cm_disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        cm_disp.plot(cmap=plt.cm.Blues)
        if path_file is None:
            path_file = self.path_models_dir.joinpath("cm_001.png")

        precision = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
        plt.title("Precision=" + str(precision))
        plt.savefig(path_file)




def read_img():
    path_current_dir = pathlib.Path(__file__).parent
    path_img = path_current_dir.joinpath("train_images", "hi", "1_10.png")

    with open(path_img, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    print(img_base64)



if __name__ == '__main__':
    ot = OnnxTester()
    ot.run()


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html