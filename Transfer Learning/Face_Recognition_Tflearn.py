import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import dlib
import cv2
import os
from tqdm import tqdm
import numpy as np
import random

predictor_path = r'./shape_predictor_68_face_landmarks.dat'
face_recog_model_path = r'./dlib_face_recognition_resnet_model_v1.dat'
image_dir = r'./Dataset'
feature_vector_train = []
feature_vector_test = []
labels_train = []
labels_test = []
num_train = 0
num_test = 0
num_faces = 50   # number of faces authorized by the system + unauthorized face
num_shots = 5

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_recog = dlib.face_recognition_model_v1(face_recog_model_path)


def get_label(key):
    label = []
    for i in range(num_faces):
        if key == i:
            label.append(1)
        else:
            label.append(0)

    print(label)
    return label


for folder_name in tqdm(os.listdir(image_dir)):
    count = 0
    folder_path = os.path.join(image_dir, folder_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        dets = detector(img, 1)
        print("Number of faces detected", len(dets))
        if len(dets) == 0 or len(dets) > 1:
            pass
        else:
            for k, d in enumerate(dets):
                print("Detection {}: Left : {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
                shape = shape_predictor(img, d)
                face_descriptor = face_recog.compute_face_descriptor(img, shape)
                # print(face_descriptor)
                # print(len(face_descriptor))
                if count < num_shots:
                    num_train += 1
                    feature_vector_train.append([face_descriptor])
                    # if folder_name == "Nikhil":
                    #     labels_train.append(get_label(0))
                    # elif folder_name == "Nikita":
                    #     labels_train.append(get_label(1))
                    # elif folder_name == "Chaitanya":
                    #     labels_train.append(get_label(2))
                    # elif folder_name == "Deep":
                    #     labels_train.append(get_label(3))
                    if folder_name == "Aamir Khan":
                        labels_train.append(get_label(0))
                    elif folder_name == "AJ Styles":
                        labels_train.append(get_label(1))
                    elif folder_name == "Anushka Sharma":
                        labels_train.append(get_label(2))
                    elif folder_name == "Bebe Rexa":
                        labels_train.append(get_label(3))
                    elif folder_name == "Biswa Kalyan Rath":
                        labels_train.append(get_label(4))
                    elif folder_name == "Brad Pitt":
                        labels_train.append(get_label(5))
                    elif folder_name == "Chris Jericho":
                        labels_train.append(get_label(6))
                    elif folder_name == "Dwayne Johnson":
                        labels_train.append(get_label(7))
                    elif folder_name == "Gigi":
                        labels_train.append(get_label(8))
                    elif folder_name == "Harvey Specter":
                        labels_train.append(get_label(9))
                    elif folder_name == "Jacqueline Fernandez":
                        labels_train.append(get_label(10))
                    elif folder_name == "Jaime Presley":
                        labels_train.append(get_label(11))
                    elif folder_name == "Jeff Bezos":
                        labels_train.append(get_label(12))
                    elif folder_name == "Jeff Hardy":
                        labels_train.append(get_label(13))
                    elif folder_name == "Margot Robbie":
                        labels_train.append(get_label(14))
                    elif folder_name == "Matt Leblanc":
                        labels_train.append(get_label(15))
                    elif folder_name == "Modi Ji":
                        labels_train.append(get_label(16))
                    elif folder_name == "Mukesh Ambani":
                        labels_train.append(get_label(17))
                    elif folder_name == "Parineeti Chopra":
                        labels_train.append(get_label(18))
                    elif folder_name == "Ranbir Kapoor":
                        labels_train.append(get_label(19))
                    elif folder_name == "Ravindra Jadeja":
                        labels_train.append(get_label(20))
                    elif folder_name == "Salman Khan":
                        labels_train.append(get_label(21))
                    elif folder_name == "Shahrukh Khan":
                        labels_train.append(get_label(22))
                    elif folder_name == "Tim Cook":
                        labels_train.append(get_label(23))
                    elif folder_name == "Virat Kohli":
                        labels_train.append(get_label(24))
                    elif folder_name == "Nikhil":
                        labels_train.append(get_label(25))
                    elif folder_name == "Nikita":
                        labels_train.append(get_label(26))
                    elif folder_name == "Chaitanya":
                        labels_train.append(get_label(27))
                    elif folder_name == "Deep":
                        labels_train.append(get_label(28))
                    elif folder_name == "Class 31":
                        labels_train.append(get_label(29))
                    elif folder_name == "Class 32":
                        labels_train.append(get_label(30))
                    elif folder_name == "Class 33":
                        labels_train.append(get_label(31))
                    elif folder_name == "Class 34":
                        labels_train.append(get_label(32))
                    elif folder_name == "Class 35":
                        labels_train.append(get_label(33))
                    elif folder_name == "Class 36":
                        labels_train.append(get_label(34))
                    elif folder_name == "Class 37":
                        labels_train.append(get_label(35))
                    elif folder_name == "Class 38":
                        labels_train.append(get_label(36))
                    elif folder_name == "Class 39":
                        labels_train.append(get_label(37))
                    elif folder_name == "Class 40":
                        labels_train.append(get_label(38))
                    elif folder_name == "Class 41":
                        labels_train.append(get_label(39))
                    elif folder_name == "Class 42":
                        labels_train.append(get_label(40))
                    elif folder_name == "Class 43":
                        labels_train.append(get_label(41))
                    elif folder_name == "Class 44":
                        labels_train.append(get_label(42))
                    elif folder_name == "Class 45":
                        labels_train.append(get_label(43))
                    elif folder_name == "Class 46":
                        labels_train.append(get_label(44))
                    elif folder_name == "Class 47":
                        labels_train.append(get_label(45))
                    elif folder_name == "Class 48":
                        labels_train.append(get_label(46))
                    elif folder_name == "Class 49":
                        labels_train.append(get_label(47))
                    elif folder_name == "Class 50":
                        labels_train.append(get_label(48))
                    else:
                        labels_train.append(get_label(49))  # For Unauthorized person
                else:
                    num_test += 1
                    feature_vector_test.append([face_descriptor])
                    # if folder_name == "Nikhil":
                    #     labels_test.append(get_label(0))
                    # elif folder_name == "Nikita":
                    #     labels_test.append(get_label(1))
                    # elif folder_name == "Chaitanya":
                    #     labels_test.append(get_label(2))
                    # elif folder_name == "Deep":
                    #     labels_test.append(get_label(3))
                    if folder_name == "Aamir Khan":
                        labels_test.append(get_label(0))
                    elif folder_name == "AJ Styles":
                        labels_test.append(get_label(1))
                    elif folder_name == "Anushka Sharma":
                        labels_test.append(get_label(2))
                    elif folder_name == "Bebe Rexa":
                        labels_test.append(get_label(3))
                    elif folder_name == "Biswa Kalyan Rath":
                        labels_test.append(get_label(4))
                    elif folder_name == "Brad Pitt":
                        labels_test.append(get_label(5))
                    elif folder_name == "Chris Jericho":
                        labels_test.append(get_label(6))
                    elif folder_name == "Dwayne Johnson":
                        labels_test.append(get_label(7))
                    elif folder_name == "Gigi":
                        labels_test.append(get_label(8))
                    elif folder_name == "Harvey Specter":
                        labels_test.append(get_label(9))
                    elif folder_name == "Jacqueline Fernandez":
                        labels_test.append(get_label(10))
                    elif folder_name == "Jaime Presley":
                        labels_test.append(get_label(11))
                    elif folder_name == "Jeff Bezos":
                        labels_test.append(get_label(12))
                    elif folder_name == "Jeff Hardy":
                        labels_test.append(get_label(13))
                    elif folder_name == "Margot Robbie":
                        labels_test.append(get_label(14))
                    elif folder_name == "Matt Leblanc":
                        labels_test.append(get_label(15))
                    elif folder_name == "Modi Ji":
                        labels_test.append(get_label(16))
                    elif folder_name == "Mukesh Ambani":
                        labels_test.append(get_label(17))
                    elif folder_name == "Parineeti Chopra":
                        labels_test.append(get_label(18))
                    elif folder_name == "Ranbir Kapoor":
                        labels_test.append(get_label(19))
                    elif folder_name == "Ravindra Jadeja":
                        labels_test.append(get_label(20))
                    elif folder_name == "Salman Khan":
                        labels_test.append(get_label(21))
                    elif folder_name == "Shahrukh Khan":
                        labels_test.append(get_label(22))
                    elif folder_name == "Tim Cook":
                        labels_test.append(get_label(23))
                    elif folder_name == "Virat Kohli":
                        labels_test.append(get_label(24))
                    elif folder_name == "Nikhil":
                        labels_test.append(get_label(25))
                    elif folder_name == "Nikita":
                        labels_test.append(get_label(26))
                    elif folder_name == "Chaitanya":
                        labels_test.append(get_label(27))
                    elif folder_name == "Deep":
                        labels_test.append(get_label(28))
                    elif folder_name == "Class 31":
                        labels_test.append(get_label(29))
                    elif folder_name == "Class 32":
                        labels_test.append(get_label(30))
                    elif folder_name == "Class 33":
                        labels_test.append(get_label(31))
                    elif folder_name == "Class 34":
                        labels_test.append(get_label(32))
                    elif folder_name == "Class 35":
                        labels_test.append(get_label(33))
                    elif folder_name == "Class 36":
                        labels_test.append(get_label(34))
                    elif folder_name == "Class 37":
                        labels_test.append(get_label(35))
                    elif folder_name == "Class 38":
                        labels_test.append(get_label(36))
                    elif folder_name == "Class 39":
                        labels_test.append(get_label(37))
                    elif folder_name == "Class 40":
                        labels_test.append(get_label(38))
                    elif folder_name == "Class 41":
                        labels_test.append(get_label(39))
                    elif folder_name == "Class 42":
                        labels_test.append(get_label(40))
                    elif folder_name == "Class 43":
                        labels_test.append(get_label(41))
                    elif folder_name == "Class 44":
                        labels_test.append(get_label(42))
                    elif folder_name == "Class 45":
                        labels_test.append(get_label(43))
                    elif folder_name == "Class 46":
                        labels_test.append(get_label(44))
                    elif folder_name == "Class 47":
                        labels_test.append(get_label(45))
                    elif folder_name == "Class 48":
                        labels_test.append(get_label(46))
                    elif folder_name == "Class 49":
                        labels_test.append(get_label(47))
                    elif folder_name == "Class 50":
                        labels_test.append(get_label(48))
                    else:
                        labels_test.append(get_label(49))  # For Unauthorized person
            count += 1

#feature_vector_train = np.load("feature_vector_train.npy")
#feature_vector_test = np.load("feature_vector_test.npy")
print()
print()
print(len(feature_vector_train))
print(len(labels_train))
# num_train = 24
# num_test = 17
feature_vector_train = np.array(feature_vector_train).reshape(num_train, 128, 1, 1)
# random.shuffle(feature_vector_train)
np.save("feature_vector_train.npy", feature_vector_train)
feature_vector_test = np.array(feature_vector_test).reshape(num_test, 128, 1, 1)
# random.shuffle(feature_vector_test)
np.save("feature_vector_test.npy", feature_vector_test)
labels_train = np.array(labels_train)
np.save("labels_train.npy", labels_train)
labels_test = np.array(labels_test)
np.save("labels_test.npy", labels_test)
print(type(feature_vector_train), feature_vector_train.shape)
print(type(labels_train), labels_train.shape)

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

print("Model Fitting:")
model.fit({"input": feature_vector_train}, {"targets": labels_train}, n_epoch=num_epochs,
          validation_set=({"input": feature_vector_test}, {"targets": labels_test}),
          snapshot_step=500, show_metric=True, run_id="face_recognition")
print("Model Fit!!")

model.save("face_recognition_{}_{}_{}shotLearning_{}faces".format(learning_rate, num_epochs, num_shots, num_faces))
print("Model saved!!")
