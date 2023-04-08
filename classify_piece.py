import json
import pathlib

import numpy as np
import onnxruntime as ort



class ClassifyPiece():
    def __init__(self, threshold:float=0.8, bgr2rgb:bool=False) -> None:
        path_current_dir = pathlib.Path(__file__).parent
        self.path_piece_onnx = path_current_dir.joinpath("models", "piece.onnx")
        path_ds_info_json = path_current_dir.joinpath("models", "dataset_info.json")

        ds_info = self._load_dataset_info(path_ds_info_json)
        self.label_to_idx = ds_info["label_to_idx"]
        self.idx_to_label = ds_info["idx_to_label"]
        self.mean = ds_info["stats"]["mean"]
        self.std = ds_info["stats"]["std"]
        
        self.threshold = threshold  # Inference thresholds
        self.bgr2rgb = bgr2rgb  # Convert BGR to RGB


    def run(self, imgs:np.ndarray, output_type:str="label", threshold:float=None) -> list:
        '''
        Classify the piece species of the n images cut out in squares. Use onnx.
        imgs : (n, 64, 64, 3)  (batch_size, height, width, channels) (0-255) (R, G, B)
        output_type : 予測結果の戻り値, "label" or "idx"
        return : Predicted pieces for n-squares(3文字で表現、頭文字は先手:b, 後手:w, 空きマスはemp (empty))
        '''
        imgs_pre = self._preprocess(imgs)
        ort_session = ort.InferenceSession(str(self.path_piece_onnx), providers=['CPUExecutionProvider'])
        outputs = ort_session.run(
            None,
            {"input": imgs_pre.astype(np.float32)},
        )
        preds = np.array(outputs[0], dtype=float)  # (81, 29) Probability of each piece
        reliability = np.max(self._softmax(preds), axis=1)
        # print(reliability.shape)
        # print(reliability)
        preds = np.argmax(preds, axis=1)  # Only take out the piece index with the highest probability
        # print(preds)
        # print(preds.dtype)

        if threshold is None:
            threshold = self.threshold

        if output_type == "label":
            preds = self._convert_idx_to_label(list(preds), list(reliability), threshold)  # Converts to 3-letter label names
        elif output_type == "idx":
            preds = self._exclude_unreliable(list(preds), list(reliability), threshold)
        else:
            pass
        return preds


    def _convert_idx_to_label(self, predicted_idx:list, reliability:list, threshold:float) -> list:
        '''
        Below the threshold is "----".
        '''
        predicted_label = []
        for idx, r in zip(predicted_idx, reliability):
            if r < threshold:
                predicted_label.append("---")
            else:
                predicted_label.append( self.idx_to_label[str(idx)] )
        return predicted_label
    

    def _exclude_unreliable(self, predicted_idx:list, reliability:list, threshold:float) -> list:
        predicted_label = []
        for idx, r in zip(predicted_idx, reliability):
            if r > threshold:
                predicted_label.append( idx )
            else:
                predicted_label.append( -1 )
        return predicted_label


    def _load_dataset_info(self, path_json:pathlib.Path) -> dict:
        with open(path_json) as f:
            d = json.load(f)
        return d


    def _softmax(self, x:np.ndarray) -> np.ndarray:
        max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max)
        sum = np.sum(exp_x, axis=1, keepdims=True)
        fx = exp_x / sum
        return fx


    def _preprocess(self, imgs:np.ndarray) -> np.ndarray:
        '''
        imgs : (n, H, W, C) (0-255) (RGB) or (BGR)
        '''
        # Convert color BGR to RGB
        if self.bgr2rgb:
            imgs = imgs[:, :, :, [2, 1, 0]]

        # nHWC -> nCHW
        imgs_pre = np.transpose(imgs, axes=[0, 3, 1, 2])
        # Normalization
        imgs_pre[:,0,:,:] = (imgs_pre[:,0,:,:] / 255 - self.mean[0]) / self.std[0]  # R
        imgs_pre[:,1,:,:] = (imgs_pre[:,1,:,:] / 255 - self.mean[1]) / self.std[1]  # G
        imgs_pre[:,2,:,:] = (imgs_pre[:,2,:,:] / 255 - self.mean[2]) / self.std[2]  # B
        return imgs_pre



if __name__ == "__main__":
    imgs_dummy = np.random.randint(0, 256, (81, 64, 64, 3))  # n, H, W, C
    
    cp = ClassifyPiece()
    list81 = cp.run(imgs_dummy, "label", threshold=0.7)
    # list81 = cp.run(imgs_dummy, "idx", threshold=0.8)
    print(len(list81))
    print(list81)
