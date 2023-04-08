import pathlib

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, precision_score

import classify_piece



class OnnxTester():
    def __init__(self) -> None:
        path_current_dir = pathlib.Path(__file__).parent
        self.path_models_dir = path_current_dir.joinpath("models")
        path_img_dir = path_current_dir.joinpath("images", "test")

        self.cp = classify_piece.ClassifyPiece()
        self.labels_list = self.set_labels_list()
        path_imgs_list = self.list_test_images(path_img_dir)
        self.true_labels, self.test_imgs = self.preprocess(path_imgs_list)


    def list_test_images(self, data_dir:pathlib.Path) -> list[pathlib.Path]:
        path_list = list(data_dir.glob('**/*.png'))
        return path_list


    def preprocess(self, list_img_path:list[pathlib.Path]):
        imgs = []
        labels_idx = []
        for p in list_img_path:
            label = p.parent.stem  # directory name

            img_b_pil = Image.open(p)  # black
            img_b = np.asarray(img_b_pil)
            imgs.append(img_b)

            if label == "em":
                idx = self.cp.label_to_idx["emp"]
                labels_idx.append(idx)  # Empty square
            else:
                idx_b = self.cp.label_to_idx["b"+label]
                idx_w = self.cp.label_to_idx["w"+label]
                labels_idx.append(idx_b)
                labels_idx.append(idx_w)

                img_w = np.rot90(img_b, 2)
                imgs.append(img_w)

        imgs_np = np.array(imgs)
        labels_np = np.array(labels_idx, dtype=np.int32)
        return labels_np, imgs_np


    def set_labels_list(self):
        labels = []
        for i in range(len(self.cp.idx_to_label)):
            labels.append(self.cp.idx_to_label[str(i)])
        return np.array(labels)


    def run(self):
        pred_idx = self.cp.run(self.test_imgs, output_type="idx", threshold=0.0)
        self.plot_confusion_matrix(self.true_labels, pred_idx, self.labels_list)

        print(pred_idx)
        print(len(pred_idx))


    def plot_confusion_matrix(self, y_true:np.ndarray, y_pred:np.ndarray, labels:np.ndarray, path_png:pathlib.Path=None) -> None:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")
        cm_disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        cm_disp.plot(cmap=plt.cm.Blues)
        if path_png is None:
            path_png = self.path_models_dir.joinpath("confusion_matrix.png")

        precision = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
        plt.title("Precision=" + str(precision))
        plt.savefig(path_png)



if __name__ == '__main__':
    ot = OnnxTester()
    ot.run()


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html