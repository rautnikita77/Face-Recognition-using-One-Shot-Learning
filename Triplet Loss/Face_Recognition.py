import tensorflow as tf
import dlib
import cv2
import os
from tqdm import tqdm
import numpy as np

predictor_path = r'./shape_predictor_68_face_landmarks.dat'
face_recog_model_path = r'./dlib_face_recognition_resnet_model_v1.dat'
image_dir = r'./Database'
feature_vector_train = []
feature_vector_test = []
labels_train = []
labels_test = []

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_recog = dlib.face_recognition_model_v1(face_recog_model_path)

for folder_name in tqdm(os.listdir(image_dir)):
    count = 0
    folder_path = os.path.join(image_dir, folder_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        dets = detector(img, 1)
        print("Number of faces detected", len(dets))
        if len(dets) == 0:
            print("No face found in image", image_name)
            exit()
        if len(dets) > 1:
            print("Error in image", image_name)
            exit()

        for k, d in enumerate(dets):
            print("Detection {}: Left : {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = shape_predictor(img, d)
            face_descriptor = face_recog.compute_face_descriptor(img, shape)
            # print(face_descriptor)
            # print(len(face_descriptor))
            if count <= len(os.listdir(folder_path))/2:
                feature_vector_train.append([face_descriptor])
                if image_name == "Nikhil":
                    labels_train.append([1, 0, 0, 0, 0])
                elif image_name == "Nikita":
                    labels_train.append([0, 1, 0, 0, 0])
                elif image_name == "Deep":
                    labels_train.append([0, 0, 1, 0, 0])
                elif image_name == "Chai":
                    labels_train.append([0, 0, 0, 1, 0])
                else:
                    labels_train.append([0, 0, 0, 0, 1])  # For Unauthorized person
            else:
                feature_vector_test.append([face_descriptor])
                if image_name == "Nikhil":
                    labels_test.append([1, 0, 0, 0, 0])
                elif image_name == "Nikita":
                    labels_test.append([0, 1, 0, 0, 0])
                elif image_name == "Deep":
                    labels_test.append([0, 0, 1, 0, 0])
                elif image_name == "Chai":
                    labels_test.append([0, 0, 0, 1, 0])
                else:
                    labels_test.append([0, 0, 0, 0, 1])  # For Unauthorized person
        count += 1

print()
print()
print(len(feature_vector_train))
print(len(labels_train))
num_train = 24
num_test = 17
feature_vector_train = np.array(feature_vector_train).reshape(num_train, 128)
feature_vector_test = np.array(feature_vector_test).reshape(num_test, 128)
labels_train = np.array(labels_train).reshape(5, num_train)
labels_test = np.array(labels_test).reshape(5, num_test)
print(type(feature_vector_train), feature_vector_train.shape)
print(type(labels_train), labels_train.shape)

num_features = 128  # output of the resnet
classes = 5  # number of faces to be recognized
n1 = 64  # number of nodes in hidden layer 1
x = tf.placeholder(tf.float64, [None, num_features])
y = tf.placeholder(tf.float64)


def forward_prop(x1):
    w1 = tf.get_variable("w1", [n1, num_features], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
    b1 = tf.get_variable("b1", [n1, 1], initializer=tf.zeros_initializer(), dtype=tf.float64)
    z1 = tf.add(tf.matmul(w1, tf.transpose(x1)), b1)
    a1 = tf.nn.softmax(z1)

    w2 = tf.get_variable("w2", [classes, n1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
    b2 = tf.get_variable("b2", [classes, 1], initializer=tf.zeros_initializer(), dtype=tf.float64)
    z2 = tf.add(tf.matmul(w2, a1), b2)
    a2 = tf.nn.softmax(z2)

    return a2


def back_prop(x1):
    prediction = forward_prop(x1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_train, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    num_epochs = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Starting training...")

        for epoch in range(num_epochs):
            epoch_loss = 0
            # for _ in range(len(feature_vector)):
            _, loss = sess.run([optimizer, cost], feed_dict={x: feature_vector_train, y: labels_train})
            epoch_loss += loss

            print((epoch+1), "completed out of", num_epochs)
            print("Epoch loss is:", epoch_loss)
            print()

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, float))
        print("Accuracy:", accuracy.eval({x: feature_vector_test, y: labels_test}))
        out = tf.argmax(prediction, 1)
        output = list(out.eval({x: feature_vector_test}))
        display_output(output)


def display_output(output):
    for pred in output:
        print(pred)


back_prop(feature_vector_train)
