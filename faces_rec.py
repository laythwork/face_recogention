import face_recognition
import cv2
import numpy as np
import os
import csv

known_face_encodings=[]
known_face_names =[]

def load_encodings(directory):
    for folder in os.listdir(directory):
        with open(directory+"//"+folder+"//data.csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                known_face_names.append(folder)
                known_face_encodings.append(row)

def faces_rec(file):
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('found_faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    # best_match_index = np.argmin(face_distances)
    # name="unknown"
    # if matches[best_match_index]:
    #     name = known_face_names[best_match_index]
print(1)
load_encodings("detected_faces")
print(2)
faces_rec("unknown//2.jpg")
print(3)