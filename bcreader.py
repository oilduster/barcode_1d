import glob
import sys
from PIL import Image
from pathlib import Path
import pyocr

import cv2
import numpy as np
import os
import numpy
from pyzbar.pyzbar import decode

from pathlib import Path
from pdf2image import convert_from_path


def barcodereader_1d_cv2(inputdir,outputdir):

    # Download:
    # https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode
    # モデルは、./models配下にsr.caffemodelとsr.prototxtを配置する
    bmodel = "./models/sr.caffemodel"
    bpro = "./models/sr.prototxt"

    files = glob.glob(inputdir)
    for file in files:

        frame = cv2.imread(file)
        frame2 = frame.copy()

        qr = cv2.barcode_BarcodeDetector(bpro, bmodel)

        # data, points, straight_qrcode = qr.detectAndDecode(frame)
        srcMat, decode_info, decoded_type, corners = qr.detectAndDecode(frame)

        # バーコード値
        for bc in decode_info:
            print('バーコード: ', bc)

        # 元画像に枠をつけて出力
        cnta = 0;
        if corners  is not None :
            for bc in corners:

                pts = np.array(([bc[0][0], bc[0][1]], [bc[1][0], bc[1][1]], [bc[2][0], bc[2][1]], [bc[3][0], bc[3][1]]),
                               np.int32)
                pts = pts.reshape((1, -1, 2))

                cv2.polylines(frame2, pts, isClosed=True, color=(0, 255, 0), thickness=8)

                cnta =cnta+1

            basename_without_ext = os.path.splitext(os.path.basename(file))[0]
            cv2.imwrite(outputdir + basename_without_ext + str(cnta) +'.png', frame2)

# テスト
barcodereader_1d_cv2("./inputs/*","./outputs/")