''' 
Recognizing people can be done using only a single picture of a person by using a neural network that is already trained on a large
dataset. Import a picture in the people folder and your good to go.
'''

import face_recognition
import cv2
import os
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import imutils

# Get a reference to webcam #0 (the default one)
#video_capture = cv2.VideoCapture(src = 'ipadress/live/amba.m3u8')
video_capture = WebcamVideoStream(src=0).start()

# load image files to recognize
known_face_encodings = []

for img in os.listdir("people"):
    img_face = face_recognition.load_image_file(('people/' + img))
    img_face_encoding = face_recognition.face_encodings(img_face)[0]
    known_face_encodings.append(img_face_encoding)

#print(known_face_encodings)

known_face_names = []

for img in os.listdir("people"):
    name = str.capitalize(str(str.split(img, '.')[0]))
    known_face_names.append(name)

print(known_face_names)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

fps = FPS().start()

while True:
    # Grab a single frame of video
    #ret, frame = video_capture.read()
    frame = video_capture.read()

    #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    #video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    #video_capture.set(cv2.CAP_PROP_FPS, 10)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    frame_resize = cv2.resize(frame, (1280, 960))
    cv2.imshow('Video', frame_resize, )

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Release handle to the webcam
#video_capture.release()
video_capture.stop()
cv2.destroyAllWindows()