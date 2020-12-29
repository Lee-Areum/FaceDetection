#필요한 library import
import numpy as np
import PIL
from PIL import Image, ImageDraw,ImageFont
import cv2
import os

class detection:
    i = 0
    def __init__(self):
        pass

    def img_show(self,image):
        #테스트용 이미지 띄우기
        cv2.imshow('image', image)  # 이미지를 띄움
        cv2.waitKey(0)  # 이미지를 띄운 상태에서 정지
        cv2.destroyAllWindows()

    def FaceDetection(self,image,type):
        # 얼굴 개수 알려주는 코드
        xml = 'haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.05,5)
        print('Number of faces detected: ' + str(len(faces)))

        # 이미지에 박스 침
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.Cut_Image(x,y,w,h,gray)

        image = self.AddPercentage(image,faces,h/10)
        if type == 1:   #실시간 일 경우 type = 1
            cv2.imshow('result', image)
        elif type == 0:    #테스트 이미지 일 경우 type = 0
            self.img_show(image)
        else:
            print("error")
            
    def ImgReSize(self, img): #이미지 크기 변경하는 변수
        return cv2.resize(img,(48,48),interpolation = cv2.INTER_AREA)

    def AddPercentage(self,img,faces,n): #각 감정의 확률을 이미지에 넣는 변수
        tag = ['angry','disgust','fear','happy','sad','surprise','neutral'] #감정 라벨들
        fontsFolder = 'Font/' #폰트가 저장된 위치
        selectedFont = ImageFont.truetype(os.path.join(fontsFolder, 'ELAND 초이스 B.ttf'), int(n))  # 폰트 경로와 사이즈 지정
        img2 = Image.fromarray(img) #배열 형태의 img를 이미지 형태로 변경
        draw = ImageDraw.Draw(img2)

        label =""
        for i in range(6): # label 생성
            label += (tag[i]+" : "+"0.0000"+"\n")
        # 이미지에 박스 침
        if len(faces):
            for (x, y, w, h) in faces: #얼굴 개수 대로 좌표를 땀
                draw.text((x-(n*7.0), y+10), label, fill="white", font=selectedFont, align='center') #글자를 더함
        img2 = np.array(img2) #array형태로 이미지 변환
        cv2.imwrite('image/addtest.jpg',img2) #이미지 저장
        return img2
            
    def Cut_Image(self,x,y,w,h,img): #이미지를 잘라서 저장하는 변수
        img_trim = img[y:y+h,x:x+w]
        resized_img = self.ImgReSize(img_trim)
        cv2.imwrite('image/cutimage/faceimage'+str(detection.i)+'.jpg',resized_img)
        detection.i += 1
        
    def TestRun(self):
        #테스트용 이미지 사용
        image = cv2.imread('image/test.jpg')
        self.FaceDetection(image, 0)

    def RealRun(self):
        cap = cv2.VideoCapture(0) #노트북 웹캠을 카메라로 사용
        cap.set(3,640) #너비
        cap.set(4,480) #높이

        while(True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) #좌우 대칭 적용

            self.FaceDetection(frame, 1)
            k = cv2.waitKey(1) & 0xff #딜레이 시간 결정

            if k ==27 : #ESC 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cImg = detection() #클래스 생성
    cImg.TestRun() #테스트 이미지용 실행
    #cImg.RealRun() #실시간 이미지용 실행



