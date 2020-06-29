import cv2 as cv
import json

def binarizeImage(imgPath):
    imgOrg = cv.flip(cv.imread(imgPath), 0)
    img = cv.cvtColor(imgOrg, cv.COLOR_BGR2GRAY)
    img = 255 - img
    ret, thresh = cv.threshold(img, 50, 255, 0)
    return imgOrg, thresh
    

def getContourHierachy(img, imgOrg):
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    img2 = cv.drawContours(imgOrg, contours, 1, (0, 255, 0), 1)
    return img2, contours, hierarchy


if __name__ == "__main__":
    imgOrg, img = binarizeImage('/home/todd/Documents/Workspace/maskProject/Data/index.png')
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

    cv.imshow('Contour', img2)
    cv.imshow('Org', img)
    cv.waitKey(0)
    cv.destroyAllWindows()