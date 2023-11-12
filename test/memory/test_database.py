import time
import numpy
from pymongo import database
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
from memory.databasewrapper import Database, Session, User, Utterance


class TestDatabase:
    def _pre(self):
        self.db: Database = Database("test_db", clear=True)

    def _post(self):
        pass

    def test_get_cue_card_random(self):
        self._pre()

        card, card_id = self.db.get_cue_card_random()
        card_copy = self.db.get_cue_card_by_id(card_id)

        assert card == card_copy

        self._post()

    def test_get_follow_up(self):
        self._pre()

        card, id = self.db.get_cue_card_random()
        follow_up = self.db.get_follow_up_by_id(id)

        assert type(follow_up) is list and len(follow_up) > 0

        self._post()

    def test_insert_get_user(self):
        self._pre()

        user = User("user", numpy.array([]))
        self.db._insert_user(user)
        other_user = self.db._get_user_by_name("user")

        assert user == other_user

        self._post()

    def test_insert_get_session(self):
        self._pre()

        user = User("user", numpy.array([]), _id=1)

        session_one = Session(user, start_time=10.0)
        session_two = Session(user, start_time=20.0)

        self.db._insert_session(session_one)
        self.db._insert_session(session_two)

        sessions = self.db.get_sessions_by_user(user)

        assert len(sessions) == 2 and (
            (sessions[0] == session_one and sessions[1] == session_two)
            or (sessions[0] == session_two and sessions[1] == session_one)
        )

        self._post()

    def test_insert_utterance(self):
        self._pre()

        utterance = Utterance([], time.time(), True)
        self.db.insert_utterance(utterance)
        found_utterance = self.db.get_last_utterance()

        assert utterance == found_utterance

        self._post()
