import cv2 as cv
import json
import subprocess
import argparse
import os
import shutil

mat_str = "newmtl baseColor\n\
Ka 1.000000 1.000000 1.000000\n\
Kd 1.000000 1.000000 1.000000\n\
map_Kd "

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

    img2 = cv.drawContours(imgOrg, contours, -1, (255, 0, 0), 3)
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

def writeMaterial(input_image, output_obj):
    material_data = mat_str + os.path.basename(input_image)
    mat_path = os.path.join(os.path.dirname(output_obj), os.path.splitext(os.path.basename(output_obj))[0]) + '.mtl'
    with open(mat_path, 'w') as matFile:
        matFile.write(material_data)
    shutil.copy(input_image, os.path.join(os.path.dirname(output_obj), os.path.basename(input_image)))
    # last need to update the obj data.
    with open(output_obj, 'r+') as objFile:
        content = objFile.read()
        objFile.seek(0, 0)
        material_reference = "mtllib " + os.path.basename(mat_path) + "\nusemtl baseColor\n"
        objFile.write(material_reference + content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LogoMaker 1.5")
    parser.add_argument("-i", "--input", help="the input image that is alpha transparent in background to create the 2.1D model", type=str)
    parser.add_argument("-j", "--json", help="the json that has the formatted hierarchy and contour info, if you already have it.", nargs='?', default = '', type=str)
    parser.add_argument("-e", "--executable", help="the executable to create the model", type=str)
    parser.add_argument("-x", "--width", help="the width of the object", type=str)
    parser.add_argument("-y", "--height", help="the height of the object", type=str)
    parser.add_argument("-t", "--thickness", help="the thickness of the object", type=str)
    parser.add_argument("-o", "--output", help="the output obj path", type=str)
    parser.add_argument("-r", "--rotateN90", help="if the object is laydown, put Y here", type=str, default="N")

    args = parser.parse_args()
    json_path = './data.json'
    if args.json == '':
        imageProcessing(args.input)
    else:
        json_path = args.json
    print("Call cpp to generate mesh.")
    if not os.path.exists(args.executable):
        print("do not work on modeling part since executable is not available.")
    exe_args = (args.executable, 'B', args.width, args.height, args.thickness, json_path, args.output, 'a1000q30', args.input, args.rotateN90) #a1000q30
    popen = subprocess.Popen(exe_args, stdout=subprocess.PIPE)
    popen.wait()
    writeMaterial(args.input, args.output)
    output = popen.stdout.read()
    print(output)