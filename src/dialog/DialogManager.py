from dialog.fsms.DialogMachine import NoDialogStateExistsException
from dialog.fsms.Dialogues import IllyDialog
from furhat.Furhat import Furhat
from dialog.facerecogniser import FaceRecogniser
from memory.memorymanager import MemoryManager
from dialog.AffectModel import AffectModel
from dialog.user_intent.UserIntentClassification import UserIntentClassification, Intent


class DialogManager:

    def __init__(self, feedback = True):
        self.memory = MemoryManager()
        self.emotion_recogniser = AffectModel()
        self.furhat = Furhat("localhost", self.memory, self.emotion_recogniser)
        voices = self.furhat.get_voices()
        self.furhat.set_voice(name="Kendra-Neural")
        self.face_recogniser = FaceRecogniser(self.memory)
        self.intent_classifier = UserIntentClassification()
        face_encoding = self.face_recogniser.get_user_face()
        self.dialog = IllyDialog(self.furhat, self.memory, face_encoding, feedback=feedback)
        self.turn_taking_policy = "auto"
        self.frustration_count = 0  # number of times the user was frustrated
        # TODO actually do something with this

    def run(self):
        """
        switch on different turn taking policies
        """
        # if self.turn_taking_policy == "auto":
        self.run_with_auto_turntaking()

    def run_with_auto_turntaking(self):

        print("Running dialog with automatic turn-taking...")

        user_speech = self.dialog.perform()

        while True:
            print(user_speech)

            if hasattr(user_speech, "emotion"):
                frustration_emotions = ["anger", "disgust", "annoyance"]
                if user_speech.emotion in frustration_emotions:
                    self.frustration_count += 1

            intent = self.get_intent(user_speech.message)
            # print(f"Intent was: {intent}")

            if intent == Intent.SILENCE:
                # if the user was silent for at least 9 seconds, ask them to speak,
                # without changing the current state of the dialog
                user_speech = self.furhat.pseudo_ask_after_silence()
                if self.get_intent(user_speech.message) == Intent.SILENCE:
                    user_speech = self.furhat.pseudo_ask_after_silence()
                    if self.get_intent(user_speech.message) == Intent.SILENCE:
                        user_speech = self.furhat.react_to_silence()
                continue

            if intent == Intent.CLARIFICATION:
                # if there is no state for the recognized intent, ask the user for clarification,
                # without changing the current state of the dialog
                user_speech = self.furhat.ask_for_clarification()

            clarification_emotions = ["confusion", "curiosity"]
            if user_speech.emotion in clarification_emotions:
                # TODO repeat the state question
                pass

            self.dialog.dialog_listen(intent)

            if self.dialog.current_state.name == "session_feedback":
                # flush short term memory
                # TODO check if this works
                self.memory.stop_session()

            user_speech = self.dialog.perform()

            if not self.dialog.has_next():
                break


        # end the session
        self.end_dialog()
        print("DIALOG ENDED")

    def end_dialog(self):
        """
            ends the dialog by closing the dialog, and the session
            closes the face recogniser
        """
        self.dialog.end()
        self.face_recogniser.close()

    def get_intent(self, text:str) -> Intent | None:
        """
            Returns the most probable intent that is possible to take in the current state
            Return Intent.SPEECH if in a speech state
            Return Intent.SILENCE if the text is empty
        """

        if self.dialog.current_state.is_speech_state:
            # never return a silence intent if we are in a speech state
            return Intent.SPEECH

        if text.strip() == "":
            return Intent.SILENCE

        possible_intents: list[Intent] = self.dialog.get_possible_intents()
        #print(f"possible intents were: {possible_intents}")
        ranked_intents: list[Intent] = self.intent_classifier.get_intents(text)
        for intent in ranked_intents:
            if intent in possible_intents:
                return intent

        return None