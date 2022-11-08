import pathlib
import pickle

import cv2
import numpy as np
import onnxruntime as ort

import classify_piece



class ClassifyPieceCamera():
    def __init__(self) -> None:
        self.init_camera()
        self.cp = classify_piece.ClassifyPiece()

    def init_camera(self):
        pass





    def camara_cap(self) -> None:
        # Webカメラ
        DEVICE_ID = 0 
        WIDTH = 640
        HEIGHT = 480
        FPS = 30
        cap = cv2.VideoCapture(DEVICE_ID)
        print(cap.isOpened())
        print("CAP")
        print(cap)


        # フォーマット・解像度・FPSの設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        # フォーマット・解像度・FPSの取得
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps:{} width:{} height:{}".format(fps, frame_width, frame_height))


        box_width = 100
        box_height = box_width * 1.1
        xc = frame_width / 2
        yc = frame_height / 2
        x0 = round(xc - box_width/2)
        y0 = round(yc - box_height/2)
        x1 = round(xc + box_width/2)
        y1 = round(yc + box_height/2)


        delay = 1
        n = 0
        while True:
            # カメラ画像取得
            _, frame = cap.read()

            # print(_)

            if frame is None:
                continue

            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0))
            frame_cropped = frame[y0:y1, x0:x1, :]

            pred = self.inference(frame_cropped)
            print(pred)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('c'):
                print("Push C key : " + str(n))
                n += 1

            elif key == ord('q'):  # 'q'をタイプされたらループから抜ける
                break
        
            else:
                pass

            # break

        # VideoCaptureオブジェクト破棄
        cap.release()
        cv2.destroyAllWindows()



    def inference(self, img:np.ndarray) -> str:
        img_preprocessed = self.preprocess(img)  # (1, 1, 64, 64)
        pred = self.cp.run(img_preprocessed)
        return pred[0]


    def preprocess(self, img, mean:float=0.0, std:float=1.0):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        img64 = self._extract_board(img_gray, [(0,0),(w,0),(0,h),(w,h)])

        img_01 = img64 / 255
        img_st = (img_01 - mean)/std

        img116464 = img_st[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

        # print(img116464)
        return img116464

    def _extract_board(self, img, board_corners) -> np.ndarray:
        '''
        盤面の抽出(切り抜き)
        input: OpenCV image, [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        return: OpenCV image
        '''
        dstSize = 64

        pts1 = np.float32(board_corners)
        pts2 = np.float32([[0,0],[dstSize,0],[0,dstSize],[dstSize,dstSize]])

        mat = cv2.getPerspectiveTransform(pts1,pts2)
        img_dst = cv2.warpPerspective(img, mat, (dstSize, dstSize))
        return img_dst



if __name__ == '__main__':
    cpc = ClassifyPieceCamera()
    cpc.camara_cap()

