import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import dlib
import cv2
import os
from tqdm import tqdm
import numpy as np
import random
from time import time


def get_label(index):
    if index == 0:
        return "Aamir Khan"
    if index == 1:
        return "AJ Styles"
    if index == 2:
        return "Anushka Sharma"
    if index == 3:
        return "Bebe Rexa"
    if index == 4:
        return "Biswa Kalyan Rath"
    if index == 5:
        return "Brad Pitt"
    if index == 6:
        return "Chris Jericho"
    if index == 7:
        return "Dwayne Johnson"
    if index == 8:
        return "Gigi"
    if index == 9:
        return "Harvey Specter"
    if index == 10:
        return "Jacqueline Fernandez"
    if index == 11:
        return "Jaime Presley"
    if index == 12:
        return "Jeff Bezos"
    if index == 13:
        return "Jeff Hardy"
    if index == 14:
        return "Margot Robbie"
    if index == 15:
        return "Matt Leblanc"
    if index == 16:
        return "Modi Ji"
    if index == 17:
        return "Mukesh Ambani"
    if index == 18:
        return "Parineeti Chopra"
    if index == 19:
        return "Ranbir Kapoor"
    if index == 20:
        return "Ravindra Jadeja"
    if index == 21:
        return "Salman Khan"
    if index == 22:
        return "Shahrukh Khan"
    if index == 23:
        return "Tim Cook"
    if index == 24:
        return "Virat Kohli"
    if index == 25:
        return "Nikhil"
    if index == 26:
        return "Nikita"
    if index == 27:
        return "Chaitanya"
    if index == 28:
        return "Deep"
    if index == 29:
        return "Unauthorized Person"


def get_key(prediction):
    index = np.argmax(prediction)
    return get_label(index)


def get_feature(frame):
    predictor_path = r'./shape_predictor_68_face_landmarks.dat'
    face_recog_model_path = r'./dlib_face_recognition_resnet_model_v1.dat'
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_recog = dlib.face_recognition_model_v1(face_recog_model_path)
    dets = detector(frame, 1)
    print("Number of faces detected", len(dets))
    if len(dets) == 0 or len(dets) > 1:
        print("No image detected")
        return [0]
    else:
        for k, d in enumerate(dets):
            print("Detection {}: Left : {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = shape_predictor(frame, d)
            face_descriptor = face_recog.compute_face_descriptor(frame, shape)
        return face_descriptor


def real_time():
    feature_vector_train = np.load("feature_vector_train.npy")
    feature_vector_test = np.load("feature_vector_test.npy")
    labels_train = np.load("labels_train.npy")
    labels_test = np.load("labels_test.npy")

    num_faces = 30
    num_features = 128  # output of the resnet
    classes = num_faces  # number of faces to be recognized
    n1 = 68  # number of nodes in hidden layer 1
    n2 = 68  # number of nodes in hidden layer 2
    n3 = 32  # number of nodes in hidden layer 3
    n4 = 16  # number of nodes in hidden layer 3
    n5 = 16  # number of nodes in hidden layer 3
    learning_rate = 0.03
    num_epochs = 100

    output = input_data(shape=[None, num_features, 1, 1], name="input")

    output = fully_connected(output, n1, activation="relu")

    # output = fully_connected(output, n2, activation="relu")
    #
    # output = fully_connected(output, n3, activation="relu")
    #
    # output = fully_connected(output, n4, activation="relu")
    #
    # output = fully_connected(output, n5, activation="relu")

    output = fully_connected(output, classes, activation="softmax")

    output = regression(output, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(output)

    print("Starting real time detection...")

    model.load("face_recognition_0.03_100_2shotLearning_30faces")

    cap = cv2.VideoCapture(0)
    previous = time()
    delta = 0

    while True:
        current = time()
        delta += current - previous
        previous = current

        if delta > 5:
            delta = 0

        ret, frame = cap.read()
        feature = np.array(get_feature(frame)).reshape(1, 128, 1, 1)

        prediction = model.predict(feature)
        output = get_key(prediction)
        cv2.imshow(output, frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


real_time()
