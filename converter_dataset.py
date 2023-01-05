import base64
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import cv2



def convert(path_img:pathlib.Path) -> dict:
    label = path_img.parent.name  # Directory name

    with open(path_img, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    d = {
        "piece":label,
        "characters":None,
        "font":None,
        "image":img_base64
    }
    return d


def read_json(path_json:pathlib.Path) -> dict:
    
    pass






if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent

    d_list = []


    path_img = path_current_dir.joinpath("train_images", "hi", "1_10.png")

    d = convert(path_img)
    print(d)

    d_list.append(d)

    with open('tmp.json', 'wt') as f:
        json.dump(d_list, f, indent=2, ensure_ascii=False)
        