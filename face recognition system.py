import cv2
import numpy as np
import face_recognition as fr

vide_capture = cv2.VideoCapture(0)
image = fr.load_image_file(r"E:\Python Project\ML-projects-\58c860a42c00002000fee66c.webp")
image_face_encoding = fr.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["dilan"]
