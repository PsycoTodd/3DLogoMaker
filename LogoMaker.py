import cv2 as cv
import json
import subprocess
import argparse
import os

def binarizeImage(imgPath):
    imgOrg = cv.flip(cv.imread(imgPath), 0)
    img = cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY)
    img = 255 - img
    ret, thresh = cv.threshold(img, 10, 255, 0)
    return imgOrg, thresh


def binarizeAlphaImage(imgPath):
    imgOrg = cv.flip(cv.imread(imgPath, cv.IMREAD_UNCHANGED), 0)
    img = imgOrg[:,:,3]
    #img = cv.cvtColor(alpha, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 10, 255, 0)
    return imgOrg, thresh
    

def getContourHierachy(img, imgOrg):
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    img2 = cv.drawContours(imgOrg, contours, -1, (0, 255, 0), 1)
    return img2, contours, hierarchy


def imageProcessing(input_path):
    imgOrg, img = binarizeAlphaImage(input_path)
    img2, contours, hierarchy  = getContourHierachy(img, imgOrg)

    data = {}
    data['hierarchy'] = []
    data['contours'] = []
    for h in hierarchy[0]:
        data['hierarchy'].append(h.tolist())
    for c in contours:
        contour = []
        for point in c:
            contour.append(point[0].tolist())
        data['contours'].append(contour)
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

    cv.imwrite('binary.png',img)

    cv.imshow('Contour', img2)
    cv.imshow('Org', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LogoMaker 1.0")
    parser.add_argument("-i", "--input", help="the input image that is binarized to create the 2.1D model", type=str)
    parser.add_argument("-e", "--executable", help="the executable to create the model", type=str)
    parser.add_argument("-o", "--output", help="the output obj path", type=str)

    args = parser.parse_args()
    imageProcessing(args.input)

    print("Call cpp to generate mesh.")
    if not os.path.exists(args.executable):
        print("do not work on modeling part.")
    args = (args.executable, 'B', './data.json', args.output, 'a1000q30', args.input) #a1000q30
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output)