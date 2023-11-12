from dialog.fsms.DialogMachine import DialogMachine, DialogState
from furhat import Furhat
from memory.memorymanager import MemoryManager
from numpy._typing import NDArray
from dialog.user_intent.UserIntentClassification import Intent


class IllyDialog(DialogMachine):

    """The Dialog carried out by Illy
    TODO interfacing the memory for storing and retrieving session data
    """

    def __init__(self, furhat: Furhat, memory: MemoryManager, image: NDArray, feedback: bool):

        self.furhat = furhat
        self.fsm = {}  # {name_of_state (String) : state (DialogState)}
        self.initial_state = None
        self.session_info = {"name": None,
                             "speech_duration": 0,
                             "user_performance": None}
        self.session_questions = {"main_q": None,
                                  "follow_up_1": None,
                                  "follow_up_2": None}

        self.memory_manager = memory
        known, name, progress = self.memory_manager.start_session(image)
        self.feedback = feedback
        print(f"Running with feedback = {self.feedback}")

        if known:
            self.session_info["name"] = name
            self.session_info["user_performance"] = progress

        self.get_questions()

        # Adding all the states and the corresponding messages to the FSM
        # for speech states, initial states and say-only states pass the correct argument when adding the state
        self.add_state("initial_state", "Hi, How are you?", memory, initial_state=True)
        self.add_state("info", "My name is Illy and I will be helping you train for your IELTS exam. What's your name?", memory, get_name=True)
        # The greet state is defined again after getting the user's name

        if feedback:
            self.add_state("greet", f"Hello {self.session_info['name']}. Would you want to start a new practice session"
                                    f" or get your progress report?", memory)
        else:
            self.add_state("greet", f"Hello {self.session_info['name']}. Would you want to start a new practice session?", memory)

        if feedback:
            self.add_state("welcome_back", f"Welcome back {self.session_info['name']}. Would you want to start a new practice session "
                                    f" or get your progress report?", memory)
        else:
            self.add_state("welcome_back",
                           f"Welcome back {self.session_info['name']}. Would you want to start a new practice session "
                           f" or get your progress report?", memory)
        self.add_state("practice", "Ok, During the practice session, I will give you a topic., "
                                   "you have to talk about this topic for two minutes. "
                                   "Then, I will ask you two follow-up questions."
                                   f" Let's start. Here goes the topic...: {self.session_questions['main_q']} ", memory,  speech_state=True)
        self.add_state("followup1", f"Now I'll ask you the first follow-up question. {self.session_questions['follow_up_1']}", memory, speech_state=True)
        self.add_state("followup2", f"Now I'll ask you the second follow-up question. {self.session_questions['follow_up_2']}", memory, speech_state=True)
        self.add_state("feedback", "I'm giving you the session feedback,"
                                           " Would you want to hear your overall progress? ", memory)
        self.add_state("progress", f"{progress} "
                                   "Do you want to start a new session?", memory)

        if feedback:
            self.add_state("new_session", "Do you want to start a new session?", memory)
        else:
            self.add_state("new_session", "That's it. Well done. Do you want to start a new session?", memory)
        self.add_state("bye", "Congrats! That's it for this session. Take care.", memory, say_only_state=True)

        # Setting the next_states for all states
        # next_states = {current_state: {intent1:next_state, intent2:next_state , }}
        next_states = {"initial_state": {
                    Intent.GREETING: "info",
               },
                "info": {
                    Intent.INTRODUCTION: "greet"
                },
                "greet": {
                    Intent.PRACTICE: "practice",
                    Intent.FEEDBACK: "progress",
                 },
                "practice": {
                    Intent.SPEECH: "followup1"
                },
                "followup1": {
                    Intent.SPEECH: "followup2"
                },
                "followup2": {
                    Intent.SPEECH: "feedback"
                },
                "feedback": {
                    Intent.CONFIRM: "progress",
                    Intent.DECLINE: "new_session"
                },
                "progress": {
                    Intent.CONFIRM: "practice",
                    Intent.DECLINE: "bye"
                },
                "new_session": {
                    Intent.DECLINE: "bye",
                    Intent.CONFIRM: "practice"
                }
        }

        if known:
            """
            If the user is known, skip getting the name, and go to welcome back
            """
            next_states["initial_state"] = {
                Intent.GREETING: "welcome_back"
            }
            next_states["welcome_back"] = {
                Intent.PRACTICE: "practice",
                Intent.FEEDBACK: "progress",
            }

        if not feedback:
            """ for the control group """
            next_states = {"initial_state": {
                Intent.GREETING: "info",
            },
                "info": {
                    Intent.INTRODUCTION: "greet"
                },
                "greet": {
                    Intent.CONFIRM: "practice",
                    Intent.DECLINE: "bye",
                },
                "practice": {
                    Intent.SPEECH: "followup1"
                },
                "followup1": {
                    Intent.SPEECH: "followup2"
                },
                "followup2": {
                    Intent.SPEECH: "new_session"
                },
                "new_session": {
                    Intent.DECLINE: "bye",
                    Intent.CONFIRM: "practice"
                }
            }


        self.set_next_states(next_states)

        super().__init__(self.fsm["initial_state"], self.furhat, self.fsm)

    def get_questions(self):
        self.session_questions["main_q"] = self.memory_manager.get_cue_card()
        self.session_questions["follow_up_1"] = self.memory_manager.get_follow_up()
        self.session_questions["follow_up_2"] = self.memory_manager.get_follow_up()

