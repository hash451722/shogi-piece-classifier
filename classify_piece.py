import pathlib
import json

import numpy as np
import onnxruntime as ort



class ClassifyPiece():
    def __init__(self) -> None:
        path_current_dir = pathlib.Path(__file__).parent
        self.path_piece_onnx = path_current_dir.joinpath("models", "piece.onnx")
        path_ds_info_json = path_current_dir.joinpath("models", "dataset_info.json")
        
        ds_info = self._load_dataset_info(path_ds_info_json)
        self.label_to_idx = ds_info["label_to_idx"]
        self.idx_to_label = ds_info["idx_to_label"]
        self.mean = ds_info["stats"]["mean"]
        self.std = ds_info["stats"]["std"]


    def run(self, img:np.ndarray, output_type:str="label") -> list:
        '''
        マス目で切り出されたn個の画像の駒種を分類する. use onnx.
        img : (n, 3, 64, 64)  (batch_size, channels, height, width) (R, G, B)
        output_type : 予測結果の戻り値, "label" or "idx"
        return : Predicted pieces for n-squares(3文字で表現、頭文字は先手:b, 後手:w, 空きマスはemp (empty))
        '''
        ort_session = ort.InferenceSession(str(self.path_piece_onnx), providers=['CPUExecutionProvider'])
        outputs = ort_session.run(
            None,
            {"input": img.astype(np.float32)},
        )
        preds = np.array(outputs[0], dtype=float)  # (81, 29) Probability of each piece
        preds = np.argmax(preds, axis=1)  # Only take out the piece index with the highest probability
        if output_type == "label":
            preds = self._convert_idx_to_label(list(preds))  # Converts to 3-letter label names
        return preds


    def _convert_idx_to_label(self, predicted_idx:list) -> list:
        predicted_label = []
        for idx in predicted_idx:
            predicted_label.append( self.idx_to_label[str(idx)] )
        return predicted_label


    def _load_dataset_info(self, path_json:pathlib.Path) -> dict:
        with open(path_json) as f:
            d = json.load(f)
        return d



if __name__ == "__main__":
    img_cells_dummy = np.random.randn(81, 3, 64, 64)  # (N, C, H, W)

    cp = ClassifyPiece()
    list81 = cp.run(img_cells_dummy, "label")
    print(len(list81))
    print(list81)