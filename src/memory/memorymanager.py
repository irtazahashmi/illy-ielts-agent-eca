from __future__ import annotations
from random import Random

import time
import numpy as np
from typing import Tuple, Dict


from numpy._typing import NDArray
from spacy.tokens.doc import Doc

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


from memory.processing.pipeline import Pipeline, PosPipe
from memory.databasewrapper import (
    Database,
    Session,
    User,
    Utterance,
)
from utils.topic_model import TopicModel


class MemoryManager:
    """Main memory interface module"""

    processing = Pipeline()
    session: Session | None = None
    user: User | None = None
    topic_model: TopicModel = TopicModel()
    face: NDArray | None = None  # user's image provided when session starts

    topic_mistakes = 0
    number_of_utterances = 0

    def __init__(self, database_name: str | None = None, clear_db: bool = False):
        self.db: Database = (
            Database(clear=clear_db)
            if database_name is None
            else Database("database_name", clear=clear_db)
        )

        # Cached processed cue card
        self._tokenized_cue_card: list[tuple[str, str]] | None = None

    # SESSION
    def start_session(self, face_encoding: NDArray):
        """
        Starts a new session

        returns a boolean for whether the user is known to the agent or not
        If the user is known, name and progress data is loaded from the database and returned
        If the user is not known, return false

        # TODO add return type when progress's structure is known
        """
        if self.session is not None:
            raise ActiveSessionException(
                "Stop active session before starting a new one"
            )

        self.number_of_utterances = 0
        self.topic_mistakes = 0

        self.face = face_encoding
        user = self.user_identify(self.face)

        name = None
        progress = None
        known = False

        if user is not None:
            # if user is known, retrieve name and progress from database
            print("User already known.")
            self.session = Session(user)

            name = self.get_user_name()
            progress = self.get_user_progress_report()
            known = True

        else:
            print("New user.")
            self.session = Session(user)

        return known, name, progress

    def stop_session(self) -> str:
        """Stops the session and flushes short-term memory to long-term"""
        if self.session is None:
            return

        self.session.end_time = time.time()
        self.session.on_topic = (
            (
                (self.number_of_utterances - self.topic_mistakes)
                / self.number_of_utterances
            )
            if self.number_of_utterances != 0
            else 1.0
        )

        self.db.flush_short_term(self.session)
        report = self.get_user_session_report()
        # self.session = None
        return report

    # USER
    def user_info(self) -> User:
        """Returns info on the user in the current session"""
        assert self.session is not None

        return self.session.user

    def user_identify(self, face_encoding: NDArray) -> User | None:
        """Identifies user based on given NDArray image. Returns None if user cannot be found."""

        return self.db.user_from_encodings(face_encoding)

    def user_new(self, face: NDArray, name: str) -> User:
        """Returns a new User object from the given NDArray image and name."""

        return User(name, face)

    def get_user_name(self) -> str:
        """returns the name of the user"""
        assert self.session is not None

        return self.session.user.name

    def set_new_user(self, name: str):
        """
        Called in case the user is not in the database
        Sets the current user of the session
        stores the user in database
        """
        assert self.session is not None
        assert self.face is not None

        current_user = self.user_new(self.face, name=name)
        self.session.user = current_user
        assert self.session.user is not None

    # DIALOG
    def submit_utterance(self, text: str, speech_state: bool = False):
        """
        Submit an utterance to short-term memory.
        Will take the text and process it through the processing pipeline.

        Submitted after each furhat.listen returns a value
        speech_state is True if we're in a speech state
        """
        assert self.session is not None

        if len(text.strip()) == 0 or not speech_state:
            return

        self.number_of_utterances += 1

        timestamp: float = time.time()
        tokens, metadata = self.processing.process(text)
        utterance = Utterance(tokens, timestamp, speech_state, metadata)

        self.db.insert_utterance(utterance)

        if not self._is_on_cue_topic():
            self.session.on_topic += 1

    def _is_on_cue_topic(self) -> bool:
        """Returns if the last speech was 'on topic' with regard to the cue card"""
        assert self.session is not None

        if self.session.cue_card_id is None:
            raise MissingCueCardException("A cue card has not been requested yet")

        assert self._tokenized_cue_card is not None

        last_utterance: Utterance | None = self.db.get_last_utterance()

        if last_utterance is not None:
            return self.topic_model.is_on_topic(
                last_utterance.tokens,
                self._tokenized_cue_card,
            )
        return True

    def submit_speech_data(self, speech_time: int, over_spoke: bool):
        """
        submit the speech-time and whether the user overspoke
        Only submitted after a speech-state
        """
        assert self.session is not None

        self.session.over_time = over_spoke

    def submit_speech_emotions(self, emotions: Dict[str, int]):
        """
        Only submitted after a speech-state

        possible emotions are:
        love, admiration, joy, approval, caring, excitement, amusement, gratitude, desire, anger, optimism, disapproval.
        grief, annoyance, pride, curiosity, neutral, disgust, disappointment, realization, fear, relief, confusion,
        remorse, embarrassment, surprise, sadness, nervousness
        """
        # TODO
        pass

    def get_cue_card(self) -> str:
        """
        Returns a cue card. If this function has already been called in a
        single session, it will return the same
        """
        assert self.session is not None

        # TODO Do smart things w.r.t asking questions and previous sessions
        if self.session.cue_card_id is None:
            card, _id = self.db.get_cue_card_random()
            self.session.cue_card_id = _id
            self._tokenized_cue_card = self.processing.process(card)[0]
        else:
            card = self.db.get_cue_card_by_id(self.session.cue_card_id)

        return card

    def get_follow_up(self) -> str:
        """Returns a follow-up question based on the cue card of the current session"""
        assert self.session is not None

        if self.session.cue_card_id is None:
            raise MissingCueCardException(
                "No value found for cue card, thus no follow up can be asked"
            )

        follow_ups = self.db.get_follow_up_by_id(self.session.cue_card_id)
        # print(follow_ups)

        rn = Random()
        follow_up: int = 0

        while follow_up in self.session.follow_ups_idx:
            follow_up = rn.randint(0, len(follow_ups) - 1)

            if len(follow_ups) == len(self.session.follow_ups_idx):
                break

        self.session.follow_ups_idx.append(follow_up)
        return follow_ups[follow_up]

    def _get_user_progress(self, user: User) -> list[tuple[Session, int, list[str]]]:
        """gets user progress"""

        sessions: list[Session] = self.db.get_sessions_by_user(user)
        sessions_progress: list[tuple[Session, int, list[str]]] = []

        for session in sessions:
            assert session.cue_card_id is not None

            cue_card = self.db.get_cue_card_by_id(session.cue_card_id)

            keywords = self.topic_model.preprocess(cue_card)
            topic = self.topic_model.get_topic_most_likely(keywords)

            session_progress = (session, topic[0], keywords)
            sessions_progress.append(session_progress)

        return sessions_progress

    def get_user_progress_report(self, window: int = 5) -> str:
        assert self.session is not None
        assert self.session.user is not None

        sessions_progress = self._get_user_progress(self.session.user)[0:window]

        if len(sessions_progress) == 0:
            return " Sorry. you don't have any progress report yet. Start practicing. "

        topic_factor = np.polyfit(
            list(map(lambda x: x[1], sessions_progress)),
            np.linspace(0, len(sessions_progress), len(sessions_progress)),
            1,
        )[0]
        fluency_factor = np.polyfit(
            list(map(lambda x: x[0].average_score, sessions_progress)),
            np.linspace(0, len(sessions_progress), len(sessions_progress)),
            1,
        )[0]
        over_time_factor = np.polyfit(
            list(map(lambda x: -1 if x[0].over_time else 1, sessions_progress)),
            np.linspace(0, len(sessions_progress), len(sessions_progress)),
            1,
        )[0]

        on_topic = (
            "It seems like you have been improving staying on topic! Keep it up! "
            if topic_factor > -0.1
            else "Try to stay more on topic. "
        )
        fluency = (
            f"You're getting more fluent! Your're increasing with about {'%.2f' % fluency_factor} points per session. "
            if fluency_factor > -0.1
            else f"It seems fluency has been decreasing in the last {window} sessions. Try to use more different words and repeat yourself less. "
        )
        over_time = (
            "You are improving staying on time! Great work! "
            if over_time_factor > -0.1
            else f"Based on the last {window} sessions, it seems like you have the tendency to talk over time. Please try to stay within the given time window. "
        )

        nervousness = "Regarding confidence and anxiety management, you are on point. Keep going. "

        return on_topic + over_time + fluency + nervousness

    def get_user_session_report(self) -> str:
        assert self.session is not None

        print("Retrieving session report")

        on_topic = f"In this session, it seemed like you stayed on-topic about {int(self.session.on_topic*100)}% of the time. "
        over_time = (
            "It seems like you went over the time limit. This is not a problem if it is a little bit, but do try to stay within the alotted time slot. "
            if self.session.over_time
            else " "
        )
        fluency_score = f"In terms of fluency, I've determined that you have a score of {int(self.session.average_score)} out of 9. "

        nervousness = "I couldn't detect any evident anxiety from your speech. Well done. "

        return on_topic + over_time + fluency_score + nervousness

    # NLP

    def extract_name(self, user_speech: str) -> str:
        """extracts the name from the user's speech"""

        # CASE 0: User just says the name
        # Check if string is a single word
        if len(user_speech.split()) == 1:
            return user_speech
        # CASE 1: check if the user indicated their name in the speech
        if "my name is" in user_speech:
            return user_speech.split("my name is")[1].split()[0]
        # Edge case: check if the user said "i'm"
        elif "i'm" in user_speech or "i am" in user_speech:
            return user_speech.split("i'm")[1].split()[0]
        # Edge case: check if the user said "i am"
        elif "i am" in user_speech:
            return user_speech.split("i am")[1].split()[0]

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        tokens = nltk.word_tokenize(user_speech)

        # CASE 2: check if the user's name is in the name database
        with open("memory/names.txt", "r") as f:

            content = f.read()
            lines = content.splitlines()
            # tokenize the names in the database
            names = []
            for line in lines:
                text_tokens = nltk.word_tokenize(line)
                # remove all elements that are '?'
                potential_names = [
                    x
                    for x in text_tokens
                    if x != "?"
                    and x != "="
                    and x != "F"
                    and x != "M"
                    and x != "1"
                    and x != ""
                    and x != "<"
                    and x != ">"
                    and x != "Z^"
                ]
                maybeName = potential_names[0]
                if len(maybeName) > 1:
                    names.append(maybeName)

            # Check if speech contains any of the names in the database
            possible_names = []
            for name in names:
                for token in tokens:
                    if name.lower() == token.lower():
                        possible_names.append(token)

            # Remove duplicates from a list
            possible_names = list(dict.fromkeys(possible_names))

        # CASE 2 (cont.): use natural processing (using filtered names)
        tagged = nltk.pos_tag(possible_names)
        result = []
        for tag in tagged:
            if tag[1] == "PERSON" or tag[1] == "NN" or tag[1] == "NNP":
                result.append(tag[0])

        if len(result) == 0:
            return "None"
        return result[0]


class ActiveSessionException(Exception):
    """Exception that indicates an operation cannot be completed because of an active session"""

    def __init__(self, args):
        super().__init__(*args)


class MissingCueCardException(Exception):
    """Exception that indicates an operation cannot be completed because a cue card has yet to be selected"""

    def __init__(self, args):
        super().__init__(*args)
