import cv2 as cv

def equalization(image):
    R, G, B = cv.split(image)

    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    return cv.merge((output1_R, output1_G, output1_B))

def mirror_horizontally(image):
    img_flip_h = cv.flip(image,1)
    return img_flip_h

def open_example_image():
    img = cv.imread("img/example.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resized = cv.resize(img,(220,220), interpolation=cv.INTER_AREA)
    return resized
    
# Altera o bilho e o contraste 
def brightness_and_contrast(image,alpha,beta):
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)