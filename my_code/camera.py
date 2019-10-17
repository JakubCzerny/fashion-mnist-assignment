import cv2
import os
import numpy as np
from detection import detect_object
import tensorflow as tf

frame_num = 0

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=10,
    varThreshold=2,
    detectShadows=False)

def camera_record(source):
    global frame_num, fgbg
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = tf.keras.models.load_model(model_path)
    video = cv2.VideoCapture(source)

    while video.isOpened():
        frame_num += 1
        ret, frame = video.read()

        if ret == True:
            # Display the resulting frame
            cv2.imshow('Camera view',frame)

            if frame_num % 2:
            # if True:
                result = detect_object(frame)
                if result is not None:
                    gray, masked, coords = result
                    cv2.imshow('Grayscale 28x28',gray)
                    cv2.imshow('Extracted object',masked)

                    # Make prediction, extract class name and find probability of maximum class
                    prediction = model.predict(gray.reshape(1,28,28,1)/255.)
                    idx = np.argmax(prediction)
                    c = classes[idx]
                    max_prob = prediction[0,idx]

                    # probability needs to be at least 50%
                    if max_prob > 0.5:
                        print(c, max_prob)
                        offset = 20
                        x, y, w, h = coords
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(frame,
                                    "Class: {:}".format(c),
                                    (offset,offset), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1, cv2.LINE_AA)
                        cv2.putText(frame,
                                    "Probability: {:.3f}".format(max_prob),
                                    (offset,2*offset), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1, cv2.LINE_AA)
                        cv2.imshow("Detected", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()


def main():
  model_path = ''
  camera_record(0, model_path)


if __name__== "__main__":
  main()
