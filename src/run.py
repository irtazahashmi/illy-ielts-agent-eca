import cv2
import face_recognition
from dialog.DialogManager import DialogManager
from memory.memorymanager import MemoryManager
from dialog.AffectModel import AffectModel

def test_name_recognition():
    # print(MemoryManager.extract_name(user_speech="hello what the fuck is going on radek"))
    print(MemoryManager.extract_name(user_speech="this is just a test to see how Jenny will behave in this environment"))

def test_affect():
    input_text = "This is just an input text, let's see what happens."
    affect = AffectModel()
    affect.update_text(input_text)
    result = affect.predict()
    print(result)

    affect.update_text("Co sie kurwa dzieje?")
    print(affect.predict())

def test_dialog(feedback: bool):
    print("Testing Dialog")
    dialog_manager = DialogManager(feedback=feedback)
    dialog_manager.run_with_auto_turntaking()


def test_face_recognition():
    # Create an object to read camera video
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if cap.isOpened() == False:
         print("Camera is unable to open.")

    # Set resolutions of frame.
    # convert from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    manager = MemoryManager()
    frame_count = 0
    id_interval = 120

    while True:
         ret, frame = cap.read()

         if ret == True:

            # Display the frame, saved in the file
            cv2.imshow("frame", frame)
            rgb_frame = frame[:, :, ::-1]

            # Press x on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord("x"):
                break
            if cv2.waitKey(1) & 0xFF == ord("i") or frame_count % id_interval == 0:
                try:
                    left, top, right, bottom = face_recognition.face_locations(rgb_frame)[0]
                    cv2.rectangle(
                        frame, (left, top), (right, bottom), (0, 0, 255), 2
                    )  # not working? # might be because the order should be top, right, bottom, left
                    user = manager.user_identify(rgb_frame)

                    if user is not None:
                        print(
                            "Found user:",
                            user.name,
                            "at location",
                        )
                    else:
                        user = manager.user_new(rgb_frame, input("Please enter a name:"))
                        manager.db.flush_short_term(user)
                        frame_count = 0
                except:
                    print("Cannot find face..")

            frame_count += 1

    # release video capture
    # and video write objects
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")

def test_fluency():
    input_string = "Hello, my name is Radek, Radek is my name. I think I like something like this or like something else."
    input_string = 'Hello, my name is Radek.'
    input_string = ''
    input_fluency = LanguageFluency(input_string)  # create a LanguageFluency object
    input_fluency.get_speaking_fluency()  # get speaking fluency
    print("Repetitions:", input_fluency.repetitions)
    print("Language category:", input_fluency.language_category)

def test_name_extraction(s: str):
    extracted_name = MemoryManager.extract_name(s)
    print(extracted_name)


if __name__ == "__main__":
    """ feedback=True to run experiments with feedback, feedback=False for the control group with no feedback"""
    test_dialog(feedback=True)
    # test_affect()
    # test_name_recognition()
    # test_dialog()
    # test_affect()
    # test_face_recognition()
    # test_fluency()
    # test_name_extraction("jesus")



##################
### example.py ###
##################
# Example from the FurhatSDK docs

# import time
# from furhat_remote_api import FurhatRemoteAPI
# import psycopg2

# # Run the database psql
# conn = psycopg2.connect(
#     host="localhost",
#     database="ielts",
#     user="ielts_user",
#     password="ielts")

# # Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
# furhat = FurhatRemoteAPI("localhost")

# # Get the voices on the robot
# voices = furhat.get_voices()

# # Set the voice of the robot
# furhat.set_voice(name="Matthew")

# # Say Hi there!
# furhat.say(text="Hi there!")

# # Play an audio file (with lipsync automatically added)
# furhat.say(
#     url="https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav", lipsync=True
# )

# # Listen to user speech and return ASR result
# result = furhat.listen()

# # Play an audio file (with lipsync automatically added)
# furhat.say(
#     url="https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav", lipsync=True
# )

# # Perform a named gesture
# furhat.gesture(name="BrowRaise")

# # Perform a custom gesture
# furhat.gesture(body={
#     "frames": [
#         {
#             "time": [
#                 0.33
#             ],
#             "params": {
#                 "BLINK_LEFT": 1.0
#             }
#         },
#         {
#             "time": [
#                 0.67
#             ],
#             "params": {
#                 "reset": True
#             }
#         }
#     ],
#     "class": "furhatos.gestures.Gesture"
#     })

# # Get the users detected by the robot
# users = furhat.get_users()
# print(users)

# # Attend the user closest to the robot
# furhat.attend(user="CLOSEST")

# # Attend a user with a specific id
# furhat.attend(userid="virtual-user-1")

# # Attend a specific location (x,y,z)
# furhat.attend(location="(0.0, 0.2, 1.0)")

# # Set the LED lights
# furhat.set_led(red=200, green=50, blue=50)
#####################
#### END EXAMPLE ####
#####################
