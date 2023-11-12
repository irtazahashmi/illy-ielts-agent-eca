
from __future__ import unicode_literals, print_function

import json
import io
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from enum import Enum

class Intent(Enum):
    """
    Intent categories used in IllyDialog:
    DECLINE: the user declines
    CONFIRM: the user accepts or confirms
    SPEECH: the user speaks for 2 minutes or answers the follow-ups
    GREETING: the user says hi
    INTRODUCTION: the user states his/her name
    PRACTICE: the user says he/she wants to start a practice session
    FEEDBACK: the user says he/she wants to get the progress report
    """
    DECLINE = "0"
    CONFIRM = "1"
    CLARIFICATION = "2"
    SPEECH = "3"
    SILENCE = "4"
    GREETING = "5"
    INTRODUCTION = "6"
    PRACTICE = "7"
    FEEDBACK = "8"


class UserIntentClassification:
    def __init__(self, path="dialog/user_intent/train_data.json"):
        self.training_data = self._load_data(path)
        self.model = self._train_model()

    def _load_data(self, path: str):
        with io.open(path) as f:
            training_data = json.load(f)
        
        return training_data


    def _train_model(self):
        nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
        nlu_engine.fit(self.training_data)
        return nlu_engine

    def _save_model(self, path: str):
        self.model.persist(path)

    def _load_model(self, path: str):
        self.model = SnipsNLUEngine.from_path(path)

    def get_intents(self, text: str):
        intents_dict = {
            "decline": Intent.DECLINE,
            "confirm": Intent.CONFIRM,
            "clarification": Intent.CLARIFICATION,
            "speech": Intent.SPEECH,
            "silence": Intent.SILENCE,
            "greeting": Intent.GREETING,
            "introduction": Intent.INTRODUCTION,
            "practice": Intent.PRACTICE,
            "feedback": Intent.FEEDBACK
        }

        predictions = self.model.get_intents(text)
        intents = [k['intentName'] for k in predictions]
        intents = [x for x in intents if x is not None]
        intents = [intents_dict[k] for k in intents]
        return intents

    @staticmethod
    def manual_intent() -> Intent:
        smth = {str(intent)[7:].lower(): intent for intent in Intent}
        response = "NOTHING"
        intents = smth.keys()

        while response not in intents:
            response = input("Please enter the desired intent ({}):".format(intents))

        return smth[response]
