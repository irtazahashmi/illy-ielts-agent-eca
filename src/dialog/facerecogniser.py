from __future__ import annotations
import time
import cv2
import face_recognition
from numpy._typing import NDArray
from memory.memorymanager import MemoryManager
from typing import Tuple



class FaceRecogniser:
    def __init__(self, memory: MemoryManager):

        self.memory = memory

        # Create an object to read camera video
        self.cap = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise CameraNotOpenedError

    def get_user_face(self) -> NDArray:
        """
            returns the face encoding as well as the image
        """
        print("Capturing your face...")
        while True:
                ret, frame = self.cap.read()

                if ret:
                    image = frame[:, :, ::-1]
                    face_encodings: list[NDArray] = face_recognition.face_encodings(image)

                    if len(face_encodings) >= 1:
                        print("Face captured.")
                        return face_encodings[0]


    def get_user_location(self) -> Tuple | None:
        """
            return's the location of user's face
            returns A tuples of found face locations in css (top, right, bottom, left) order
            returns None if the face was not found
        """
        image = self.get_user_image()
        top, right, bottom, left = face_recognition.face_locations(image)[0]
        print(f"Face found at top: {top}, right: {right}, bottom: {bottom}, left: {left}")

        return top, right, bottom, left

    def close(self):
        """ Closes the resources """
        # release video capture
        # and video write objects
        self.cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()


class CameraNotOpenedError(Exception):
    """'Runtime' error for the camera not opening"""

    pass
