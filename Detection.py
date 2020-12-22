#필요한 library import
import cv2

class detection:
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

        if type == 1:   #실시간 일 경우 type = 1
            cv2.imshow('result', image)
        elif type == 0:    #테스트 이미지 일 경우 type = 0
            self.img_show(image)
        else:
            print("error")

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
            k = cv2.waitKey(30) & 0xff

            if k ==27 : #ESC 키를 누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cImg = detection() #클래스 생성
    cImg.TestRun() #테스트 이미지용 실행
    # cImg.RealRun() #실시간 이미지용 실행



