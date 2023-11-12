from typing import Callable
from furhat import Furhat
from typing import Dict, List
from dialog.user_intent.UserIntentClassification import Intent
from memory.memorymanager import MemoryManager


class DialogState:
    """
    Current dialog state
    """

    def __init__(self, name: str, performance: Callable, is_speech_state=False):
        self.performance = performance

        self.next: dict[Intent, DialogState] = {
            intent: None for intent in Intent
        }

        self.name = name
        self.is_speech_state = is_speech_state

    def to(self, next: "DialogState", intent: Intent) -> Intent:
        """ Set next dialog based on intent of the user """
        # rewrites the next with same intent for the name state to work properly
        self.next[intent] = next
        return intent


class DialogMachine:
    """Class that performs the dialog"""

    def __init__(self, initial_state : DialogState, furhat : Furhat, fsm_dict : Dict[str,DialogState]):

        if initial_state is None:
            raise ValueError(
                "Initial state cannot be None, please define a variable \
                initial_state"
            )

        self.current_state: DialogState = initial_state
        self.furhat = furhat
        self.fsm = fsm_dict

    def perform(self):
        """ Performs the current state and returns the user's speech"""
        user_speech = self.current_state.performance()
        return user_speech

    def dialog_listen(self, intent: Intent) -> DialogState:
        """
            Listen to the user's intent and act accordingly
        """

        next = self.current_state.next[intent]

        if type(next) is NoDialogState:
            raise NoDialogStateExistsException()

        self.current_state = next
        return self.current_state

    def has_next(self) -> bool:
        return (
            len(
                [
                    option
                    for option in self.current_state.next.values()
                    if type(option) is not NoDialogState
                ]
            )
            > 0
        )

    def add_state(self, name:str, message:str, memory_manager:MemoryManager, feedback_state=True, progress_state=True, say_only_state=False, initial_state=False, speech_state=False, get_name=False):
        """"
            Adding a state to the fsm, indicating if this is the initial state or final state]
            For speech states, self.furhat.say should receive speech_state = True argument
            If get_name = True, get the name of the user, send the name to the memory manager, and redefine greeting state
            If the state already exists, rewrites it
        """
        if name not in self.fsm.keys():
            if name == "followup2":
                # at the end of this state, get session report and remake feedback state.
                state = DialogState(name, lambda: self.followup2_f(message), is_speech_state=speech_state)
            elif name == "feedback":
                state = DialogState(name, lambda: self.feedback_f(message), is_speech_state=speech_state)
            elif say_only_state:
                state = DialogState(name, lambda: self.furhat.say(text=message), is_speech_state=speech_state)
            elif get_name:
                state = DialogState(name, lambda: self.get_name_state_f(memory_manager, text=message), is_speech_state=speech_state)
            elif speech_state:
                state = DialogState(name, lambda: self.furhat.ask(text=message, speech_state=True), is_speech_state=speech_state)
            else:
                state = DialogState(name, lambda: self.furhat.ask(text=message), is_speech_state=speech_state)

            self.fsm[name] = state
            if initial_state:
                self.set_initial_state(name)
        else:
            raise DuplicateStateException

    def set_next_states(self, nexts: Dict[str, Dict[Intent, str]]):
        """
        Setting the next_states for all states
        next_states = {current_state: {intent1:next_state, intent2:next_state , }}
        """
        for curr_state, v in nexts.items():
            for intent, next_state in v.items():
                try:
                    self.fsm[curr_state].to(self.fsm[next_state], intent)
                except KeyError:
                    raise NoDialogStateExistsException

    def set_initial_state(self, name: str):
        self.initial_state = self.fsm[name]

    def end(self):
        """
        Ends the dialog, closes the session
        """
        self.memory_manager.stop_session()

    def get_name_state_f(self, memory_manager:MemoryManager, text: str) -> str:
        """
        function for the state that receives a name from the user
            ask the user the text message
            get the name of the user
            store the name of the user
            redefine greeting state
        """
        user_speech = self.furhat.ask(text=text)
        name = memory_manager.extract_name(user_speech.message)
        memory_manager.set_new_user(name)
        if self.feedback:
            new_message = f"Hello {name}]. Would you want to start a new practice session or get your progress report?"
            progress_message = self.furhat.memory.get_user_progress_report()
        else:
            new_message = f"Hello {name}]. Would you want to start a new practice session?"

        self.fsm["greet"] = DialogState("greet", lambda: self.furhat.ask(text=new_message))
        self.fsm["info"].to(self.fsm["greet"], Intent.INTRODUCTION)

        if self.feedback:
            self.fsm["greet"].to(self.fsm["practice"], Intent.PRACTICE)
            self.fsm["greet"].to(self.fsm["progress"], Intent.FEEDBACK)

            self.fsm["progress"] = DialogState("progress", lambda: self.furhat.ask(
                text=progress_message + "Do you want to start a new session?"))
            self.fsm["feedback"].to(self.fsm["progress"], Intent.CONFIRM)
            self.fsm["greet"].to(self.fsm["progress"], Intent.FEEDBACK)
            self.fsm["progress"].to(self.fsm["practice"], Intent.CONFIRM)
            self.fsm["progress"].to(self.fsm["bye"], Intent.DECLINE)
        else:
            self.fsm["greet"].to(self.fsm["practice"], Intent.CONFIRM)
            self.fsm["greet"].to(self.fsm["bye"], Intent.DECLINE)

        return user_speech

    def get_possible_intents(self) -> list[Intent]:
        possible_intents: list[Intent] = []
        for intent, next_state in self.current_state.next.items():
            if next_state is not None:
                possible_intents.append(intent)
        return possible_intents

    def followup2_f(self,text:str):
        """
        Ask the second follow up question, get session feedback and remake the feedback state
        """
        #print("IN FOLLOWUP2")
        user_speech = self.furhat.ask(text=text, speech_state=True)
        if self.feedback:
            self.furhat.memory.stop_session()
            message = self.furhat.memory.get_user_session_report()
            self.fsm["feedback"] = DialogState("feedback", lambda: self.feedback_f(text=message + "Would you want to hear your overall progress? "))
            self.fsm["followup2"].to(self.fsm["feedback"], Intent.SPEECH)
            self.fsm["feedback"].to(self.fsm["progress"], Intent.CONFIRM)
            self.fsm["feedback"].to(self.fsm["new_session"], Intent.DECLINE)
        return user_speech

    def feedback_f(self, text: str):
        """
        provide the feedback, get progress report and remake the progress state
        """
        #print("IN FEEDBACK")
        user_speech = self.furhat.ask(text=text, speech_state=False)
        #self.furhat.memory.stop_session()
        message = self.furhat.memory.get_user_progress_report()
        self.fsm["progress"] = DialogState("progress", lambda: self.furhat.ask(text=message + " Do you want to start a new session?"))
        self.fsm["feedback"].to(self.fsm["progress"], Intent.CONFIRM)
        self.fsm["greet"].to(self.fsm["progress"], Intent.FEEDBACK)
        self.fsm["progress"].to(self.fsm["practice"], Intent.CONFIRM)
        self.fsm["progress"].to(self.fsm["bye"], Intent.DECLINE)
        return user_speech


class NoDialogState(DialogState):
    """Represents the lack of a dialog option"""

    def __init__(self):
        super()


class DialogStateExistsException(Exception):
    """'Init/compile-time' error that indicates a Dialog was constructed \
    improperly"""

    pass


class NoDialogStateExistsException(Exception):
    """'Runtime' error that indicates an dialog option"""

    pass

class DuplicateStateException(Exception):
    def __init__(self):
        """
        Raised when add_state is called with the name of a state that already exists
        """
        pass

