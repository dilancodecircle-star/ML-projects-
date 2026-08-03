import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8m-face.pt")
photo = cv2.imread("Woman-cuddling-with-cat-1536x864.jpg")
result = model(photo)
face_result = face_model(photo)
cat_person = result[0].plot()
cat_person_face = face_result[0].plot(img = cat_person)


cv2.imshow("my window" , result[0].plot())
cv2.waitKey(5000)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8m-face.pt")
while cv2.waitKey(1)!=ord("x"):
    _, frame = cap.read()
    result = model(frame , verbose = False)
   # face_result = face_model(frame , verbose = False )
    cat_person = result[0].plot()
   # cat_person_face = face_result[0].plot(img = cat_person)

    cv2.imshow("my window" , cat_person)
    
cv2.waitKey(5000)
cap.release()
cv2.destroyAllWindows()