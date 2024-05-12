import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Return 128D features for single image
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", "Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = None
        logging.warning("no face")
    return face_descriptor

# Return the mean value of 128D face descriptor for person X
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    # Filter out non-directory files such as .DS_Store
    photos_list = [f for f in os.listdir(path_face_personX) if os.path.isfile(os.path.join(path_face_personX, f))]
    if photos_list:
        for photo in photos_list:
            # Process each photo
            photo_path = os.path.join(path_face_personX, photo)
            logging.info("%-40s %-20s", "/ Reading image:", photo_path)
            features_128d = return_128d_features(photo_path)
            if features_128d is not None:
                features_list_personX.append(features_128d)
    else:
        logging.warning("Warning: No images in %s", path_face_personX)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128)
    return features_mean_personX

def main():
    logging.basicConfig(level=logging.INFO)
    # Get the order of latest person
    person_list = [person for person in os.listdir(path_images_from_camera) if os.path.isdir(os.path.join(path_images_from_camera, person))]
    person_list.sort()

    with open("/Users/admin/Downloads/Face-Recognition-Based-Attendance-System-main/data/features_all.csv", "w",
              newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            logging.info("%sperson_%s", path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(os.path.join(path_images_from_camera, person))

            # Write the person's name and the 128D features into the CSV
            writer.writerow([person] + list(features_mean_personX))
            logging.info('\n')
        logging.info("Save all the features of faces registered into: data/features_all.csv")

if __name__ == '__main__':
    main()
