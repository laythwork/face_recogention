from PIL import Image
import face_recognition
import os
from numpy import savetxt


def detect_faces(directory):
    i=0
    faces_encodings=[]
    for filename in os.listdir(directory):

        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file(directory+"//" + filename)

            # Find all the faces in the image using the default HOG-based model.
            # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
            # See also: find_faces_in_picture_cnn.py
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            for face_encoding in face_encodings:
                faces_encodings.append(face_encoding)

            print("filename " + filename)
 
            savePath="detected_faces//"+directory
            
            if not os.path.exists(savePath):
                os.makedirs(savePath)
 
            for face_location in face_locations:
                i=i+1
                # Print the location of each face in this image
                top, right, bottom, left = face_location
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                 
                # You can access the actual face itself like this:
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image= pil_image.resize((255, 255), resample=0) 
                pil_image.save(savePath +"//"+ str(i) +".jpg")                
        else:
            continue

    savetxt(savePath+'//data.csv', faces_encodings, delimiter=',')

detect_faces("ben")


