import numpy as np
from PIL import Image, ImageDraw,ImageFont
import cv2
import os

target_image = Image.open('image/test.jpg') #기본 배경 이미지 open

fontsFolder = 'Font/'
selectedFont = ImageFont.truetype(os.path.join(fontsFolder,'ELAND 초이스 B.ttf'),20) #폰트 경로와 사이즈 지정
draw = ImageDraw.Draw(target_image)
draw.text((0,0),"test 글자",fill ="white",font = selectedFont,align = 'center')
target_image.save('image/addtest.jpg')

img = target_image
# img = cv2.cvtColor(np.float32(img),cv2.COLOR_BGR2GRAY)
img = np.array(img)

cv2.imshow('image', img)  # 이미지를 띄움
cv2.waitKey(0)  # 이미지를 띄운 상태에서 정지
cv2.destroyAllWindows()
