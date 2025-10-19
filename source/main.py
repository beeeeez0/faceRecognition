import numpy as np
import cv2 as cv
import os

_CATS = f"{os.getcwd()}/data/cats"
cascadeFile = cv.CascadeClassifier(f"{os.getcwd()}/data/haarcascade_frontalcatface.xml")

def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
 
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break

        cats = cascadeFile.detectMultiScale(ret)

        if len(cats) == 0:
            print("No faces found")
 
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()