from typing import Dict
from furhat_remote_api import FurhatRemoteAPI
import time
from threading import Timer
from dialog.user_intent.UserIntentClassification import Intent
from memory.memorymanager import MemoryManager
from dialog.AffectModel import AffectModel


class Furhat(FurhatRemoteAPI):
    def __init__(self, host, memory: MemoryManager, emotion: AffectModel):
        super().__init__(host)
        self.silence_policy = "simple"
        self.clarification_policy = "simple"
        self.memory = memory
        self.emotion = emotion

        # used in speech states
        self.interrupt_time = 0
        self.interrupted = False
        self.underspoke = False

    def listen_and_submit(self, speech_state:bool = False):
        """
            listen and submit to memory if not empty
        """
        user_speech = self.listen()
        if user_speech.message.strip() != "":
            user_speech.message = user_speech.message + " "
            #print(f"Submitting Utterance: {user_speech.message}")
            self.memory.submit_utterance(user_speech.message, speech_state)
        return user_speech

    def ask(self, text: str, speech_state=False):
        """
        Stop listening, say, and start listening
        If in speech state:
            The user can speak up to two minutes
            The overall speech time of the user is calculated
            The user is interrupted if gone over two minutes
            Detecting if the user speaks for less than two minutes is done by detecting a pause of end_of_speech_pause

        user_speech = {"message" : User's message,
                        "speech_time" : Total time of user's speech if in speech state,
                        "over_spoke" : a boolean for indicating whether the user spoke over the time limit
                                        if in speech state,
                        ....}
        """

        self.listen_stop()
        say_result = self.say(text=text, blocking=True)
        if not speech_state:
            user_speech = self.listen_and_submit(speech_state)
            emotion = self.emotion.predict(user_speech.message)
            user_speech.emotion = emotion
        else:
            print("In a speech state")
            end_of_speech_pause = 2 # This is in addition to the timeout of listen
            speech_limit = 120
            start_silence_time = None
            speech = ""
            emotions: Dict[str, int] = {}  # counting the number of times different emotions were encountered
            silence = False
            start_time = time.time()
            t = Timer(speech_limit, self.interrupt)
            t.start()
            while not self.interrupted:
                user_speech = self.listen_and_submit(speech_state)
                emotion = self.emotion.predict(user_speech.message)
                if len(emotions) == 0 or emotion not in emotions.keys():
                    emotions[emotion] = 1
                else:
                    emotions[emotion] += 1
                #print(f"emotion: {emotions}")
                speech += user_speech.message

                while user_speech.message.strip() == "" and \
                        (not silence or time.time() - start_silence_time < end_of_speech_pause):
                    if not silence:
                        start_silence_time = time.time()
                        silence = True
                if silence:
                    # spoken for less than speech time
                    self.underspoke = True
                    self.interrupt_time = time.time()
                    self.say(text= "Ok...",blocking = True)
                    break

            user_speech.message = speech
            user_speech.speech_time = self.interrupt_time - start_time
            user_speech.over_spoke = self.interrupted
            user_speech.emotion = emotions

            self.memory.submit_speech_data(user_speech.speech_time, user_speech.over_spoke)
            self.memory.submit_speech_emotions(user_speech.emotion)

            print(f"End of speech. results: \n speech: {user_speech.message} \nspeech_time: {user_speech.speech_time}\n"
                  f"over_spoke: {user_speech.over_spoke}")

            print(f"Emotions dict of the speech: {user_speech.emotion}")

        return user_speech

    def pseudo_ask_after_silence(self):
        """
            Similar to the ask function
            is called when 3 seconds of silence is detected.
            Used to extend the total possible pause before Furhat reacts to silence
            This function will be called only in non-speech states
        """
        user_speech = self.listen_and_submit()
        emotion = self.emotion.predict(user_speech.message)
        user_speech.emotion = emotion

        return user_speech

    def interrupt(self):
        if self.underspoke:
            return
        else:
            self.interrupt_time = time.time()
            self.interrupted = True
            print("Interrupting...")
            self.say(text="Ok...", blocking=True)


    def ask_for_clarification(self):
        """
        Simplest method for asking for clarification, always ask the same thing
        TODO Change based on the policy if multiple policies are added
        TODO add a specific clarification for each state
        """
        clarification_message = "Sorry, I couldn't catch that. Could you please repeat?"
        user_speech = self.ask(text=clarification_message)
        return user_speech

    def react_to_silence(self):
        """
        called if the user was silent for at least 9 seconds
        Simplest method for asking for clarification, always say the same thing
        TODO to if the user answers anything, repeating the current state
        """
        silence_message = "Are you still there?"
        user_speech = self.ask(text=silence_message)
        return user_speech
