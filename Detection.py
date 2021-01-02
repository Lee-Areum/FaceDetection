#필요한 library import
import numpy as np
import PIL
from PIL import Image, ImageDraw,ImageFont
import cv2
from tensorflow.keras.models import load_model
import os

class detection:
    i = 0
    tag = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # 감정 라벨들

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

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #gray scale로 변환
        faces = face_cascade.detectMultiScale(gray,1.05,5)
        print('Number of faces detected: ' + str(len(faces))) #얼굴 개수 출력

        # 이미지에 박스 침
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cutimg = self.Cut_Image(x,y,w,h,gray) #얼굴모양으로 자르는 함수
                print('cuting image')
                image = self.AddPercentage(image,cutimg,x,y,h/10) #각 tag별 확률을 적는 함수
                print('add percentage')

        if type == 1:   #실시간 일 경우 type = 1
            cv2.imshow('result', image)
        elif type == 0:    #테스트 이미지 일 경우 type = 0
            self.img_show(image) 
        else:
            print("error")

    def predict_image(self, image): #이미지 별 클래스를 예측하는 함수
        image = np.array(image, dtype='float32')
        image /= 255
        model = load_model('learned_model_10.h5') #h5 로드
        print("end 전처리")
        pred_array = model.predict(image) #predict 호출
        print("end prediction")
        return pred_array

    def AddPercentage(self,img, cutimg,x,y,n): #각 감정의 확률을 이미지에 넣는 변수
        print("addpercentage")
        cutimg = np.stack((cutimg,)*3, axis = -1)
        cutimg = cv2.resize(cutimg, (48,48))
        cutimg = cutimg.reshape(1,48,48,3)
        pred_array = self.predict_image(cutimg)
        result = np.argmax(pred_array) #가장 확률이 높은 결과 저장
        score = float("%0.2f" % (max(pred_array[0]) * 100))  #페센트 저장
        print(f'Result: {result}, Score: {score}')
        fontsFolder = 'Font/' #폰트가 저장된 위치
        selectedFont = ImageFont.truetype(os.path.join(fontsFolder, 'ELAND 초이스 B.ttf'), int(n))  # 폰트 경로와 사이즈 지정
        img2 = Image.fromarray(img) #배열 형태의 img를 이미지 형태로 변경
        draw = ImageDraw.Draw(img2)

        print("라벨 그리기 시작")
        label =""
        for i in range(7): # label 생성
            label += (self.tag[i]+" : "+str(float("%0.2f" % (pred_array[0][i] * 100)))+"\n")
            print(f'predict :{self.tag[i]} score : {str(float("%0.2f" % (pred_array[0][i] * 100)))}')
        # 이미지에 박스 침
        draw.text((x-(n*7.0), y+10), label, fill="white", font=selectedFont, align='center') #글자를 더함
        print('rute : EMOJI/'+str(result)+'.png')
        emoji = cv2.imread('EMOJI/'+str(result)+'.png')
        emoji = cv2.resize(emoji,(70,70),interpolation = cv2.INTER_AREA)
        emoji = Image.fromarray(emoji)

        print("이모지 open")
        area = (int(n)+x,y-70)
        print(f'area x:{x} y: {y-int(n)}')
        img2.paste(emoji,area)
        print('paste 완료')

        img2 = np.array(img2) #array형태로 이미지 변환
        cv2.imwrite('image/addtest.jpg',img2) #이미지 저장
        return img2

    def ImgReSize(self, img): #이미지 크기 변경하는 변수
        return cv2.resize(img,(48,48),interpolation = cv2.INTER_AREA)

    def Cut_Image(self,x,y,w,h,img): #이미지를 잘라서 저장하는 변수
        img_trim = img[y:y+h,x:x+w]
        resized_img = self.ImgReSize(img_trim)
        cv2.imwrite('image/cutimage/faceimage'+str(detection.i)+'.jpg',resized_img)
        detection.i += 1
        return resized_img
        
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



