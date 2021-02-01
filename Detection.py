#필요한 library import
import numpy as np
import glob
import PIL
from PIL import Image, ImageDraw,ImageFont
import cv2
from tensorflow.keras.models import load_model
import os
import dlib
import time

# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

class detection:
    i = 0
#    tag = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # 감정 라벨들
    tag = ['angry', 'disgust', 'happy', 'sad', 'surprise', 'neutral']  # 감정 라벨들

    def __init__(self):
        pass

    def img_show(self,image):
        #테스트용 이미지 띄우기
        cv2.imshow('image', image)  # 이미지를 띄움
        cv2.waitKey(0)  # 이미지를 띄운 상태에서 정지
        cv2.destroyAllWindows()

    def FaceDetection(self,image,type):
        # 얼굴 개수 알려주는 코드
        """ dlib 사용전
        xml = 'haarcascades/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xml)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #gray scale로 변환
        faces = face_cascade.detectMultiScale(gray,1.05,5)
        print('Number of faces detected: ' + str(len(faces))) #얼굴 개수 출력
        """
        detector = dlib.get_frontal_face_detector()
        #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # resize the video

        #image = cv2.resize(image,dsize=(640, 480), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces (up-sampling=1)
        face_detector = detector(img_gray, 1)
        # the number of face detected
        print("The number of faces detected : {}".format(len(face_detector)))

        for face in face_detector:
            # face wrapped with rectangle
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                          (0, 0, 255), 3)
            cutimg = self.Cut_Image(face.left(),face.top(),face.right(),face.bottom(),img_gray) #얼굴모양으로 자르는 함수
#                print('cuting image')
            image = self.AddPercentage(image,cutimg,face.left(),face.top(),face.bottom()/10) #각 tag별 확률을 적는 함수
#                print('add percentage')

            # make prediction and transform to numpy array
            # landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기
            #
            # # create list to contain landmarks
            # landmark_list = []
            #
            # # append (x, y) in landmark_list
            # for p in landmarks.parts():
            #     landmark_list.append([p.x, p.y])
            #     cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

        # 이미지에 박스 침
        """ dlib 사용전
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cutimg = self.Cut_Image(x,y,w,h,gray) #얼굴모양으로 자르는 함수
#                print('cuting image')
                image = self.AddPercentage(image,cutimg,x,y,h/10) #각 tag별 확률을 적는 함수
#                print('add percentage')
        """
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
#        print("end 전처리")
        pred_array = model.predict(image) #predict 호출
#        print("end prediction")
        return pred_array

    def AddPercentage(self,img, cutimg,x,y,n): #각 감정의 확률을 이미지에 넣는 변수
#        print("addpercentage")
        cutimg = np.stack((cutimg,)*3, axis = -1)
        cutimg = cv2.resize(cutimg, (48,48))
        cutimg = cutimg.reshape(1,48,48,3)
        pred_array = self.predict_image(cutimg)
        result = np.argmax(pred_array) #가장 확률이 높은 결과 저장
        score = float("%0.2f" % (max(pred_array[0]) * 100))  #페센트 저장
        # if(score < 55):
        #     result = 5
 #       print(f'Result: {result}, Score: {score}')
        fontsFolder = 'Font/' #폰트가 저장된 위치
        selectedFont = ImageFont.truetype(os.path.join(fontsFolder, 'ELAND 초이스 B.ttf'), int(n/2))  # 폰트 경로와 사이즈 지정
        img2 = Image.fromarray(img) #배열 형태의 img를 이미지 형태로 변경
        draw = ImageDraw.Draw(img2)

#        print("라벨 그리기 시작")
        label =""
        for i in range(6): # label 생성
            label += (self.tag[i]+" : "+str(float("%0.2f" % (pred_array[0][i] * 100)))+"\n")
#            print(f'predict :{self.tag[i]} score : {str(float("%0.2f" % (pred_array[0][i] * 100)))}')
        # 이미지에 박스 침
        draw.text((x-(n*3.3), y+8), label, fill="white", font=selectedFont, align='center') #글자를 더함
 #       print('rute : EMOJI/'+str(result)+'.png')
        emoji = cv2.imread('EMOJI/'+
                           str(result)+'.png')
        emoji = cv2.resize(emoji,(70,70),interpolation = cv2.INTER_AREA)
        emoji = Image.fromarray(emoji)

#        print("이모지 open")
        area = (int(n)+x,y-70)
#        print(f'area x:{x} y: {y-int(n)}')
        img2.paste(emoji,area)
#        print('paste 완료')

        img2 = np.array(img2) #array형태로 이미지 변환
        cv2.imwrite('image/cutimage/faceimage'+str(detection.i)+'.jpg',img2) #이미지 저장
        detection.i += 1
        return img2

    def ImgReSize(self, img): #이미지 크기 변경하는 변수
        return cv2.resize(img,(48,48),interpolation = cv2.INTER_AREA)

    def Cut_Image(self,x,y,w,h,img): #이미지를 잘라서 저장하는 변수
        img_trim = img[y:y+h,x:x+w]
        resized_img = self.ImgReSize(img_trim)
        return resized_img
        
    def TestRun(self):
        #테스트용 이미지 사용
        path = glob.glob('image/Test/*.jpg')
        # image = cv2.imread('image/test.jpg')
        for img in path:
            image = cv2.imread(img)
            self.FaceDetection(image, 0)

    def RealRun(self):
        cap = cv2.VideoCapture(0) #노트북 웹캠을 카메라로 사용
 #       cap.set(3,1280) #너비
  #      cap.set(4,960) #높이
        prevTime = 0  # 이전 시간을 저장할 변수
        while(True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) #좌우 대칭 적용

            self.FaceDetection(frame, 1)

            ########### 추가 ##################
            # 현재 시간 가져오기 (초단위로 가져옴)
            curTime = time.time()

            # 현재 시간에서 이전 시간을 빼면?
            # 한번 돌아온 시간!!
            sec = curTime - prevTime
            # 이전 시간을 현재시간으로 다시 저장시킴
            prevTime = curTime

            # 프레임 계산 한바퀴 돌아온 시간을 1초로 나누면 된다.
            # 1 / time per frame
            fps = 1 / (sec)

            # 디버그 메시지로 확인해보기
            print("Time {0} ".format(sec))
            print("Estimated fps {0} ".format(fps))

            # 프레임 수를 문자열에 저장
            str = "FPS : %0.1f" % fps

            # 표시
            cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            ###################################

            k = cv2.waitKey(1)& 0xff #딜레이 시간 결정

            if k ==27 : #ESC 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cImg = detection() #클래스 생성
    #cImg.TestRun() #테스트 이미지용 실행
    cImg.RealRun() #실시간 이미지용 실행



