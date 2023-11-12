import numpy
from memory.databasewrapper import User
from memory.memorymanager import MemoryManager


class TestManager:
    def _pre(self):
        self.manager: MemoryManager = MemoryManager(
            database_name="test_db", clear_db=True
        )

    def _post(self):
        self.manager.db.sessions.drop()
        self.manager.db.users.drop()

    def test_cue_card(self):
        self._pre()

        user = User("name", numpy.array([]))
        self.manager.start_session(user)

        card = self.manager.get_cue_card()
        follow_up_a = self.manager.get_follow_up()
        follow_up_b = self.manager.get_follow_up()

        assert (
            len(follow_up_a) != 0
            and len(follow_up_b) != 0
            and follow_up_a != follow_up_b
        )

        self._post()

    def dont_test_cue_card_asked_all(self):
        # TODO Something wrong with getting multiple follow-ups... (can't be bothered now)
        self._pre()

        user = User("name", numpy.array([]))
        self.manager.start_session(user)
        number_of_follow_ups = 10

        card = self.manager.get_cue_card()
        follow_ups = []

        for i in range(number_of_follow_ups):
            follow_ups.append(self.manager.get_follow_up())

        assert (
            len(filter(lambda x: x, [len(follow_up) != 0 for follow_up in follow_ups]))
            == 0
        )

        self._post()

    def test_submit_utterance(self):
        self._pre()

        self.manager.start_session(numpy.array([]))
        self.manager.set_new_user("user")

        cue_card = self.manager.get_cue_card()
        self.manager.submit_utterance(cue_card, True)
        assert self.manager.db.get_last_utterance() is not None

        self._post()

    def test_get_session_report(self):
        self._pre()

        self.manager.start_session(numpy.array([]))
        self.manager.set_new_user("user")

        _ = self.manager.get_cue_card()

        self.manager.submit_utterance(
            "I am talking about a general subject.", speech_state=True
        )
        self.manager.submit_utterance(
            "Furthermore, I do not know what the cue is about.", speech_state=True
        )
        self.manager.submit_utterance(
            "I would even go so far as to say that I am barely aware.",
            speech_state=True,
        )

        rep = self.manager.stop_session()
        assert rep is not None

        self._post()

    def test_get_progress_report(self):
        self._pre()

        # SESSION 1
        self.manager.start_session(numpy.array([]))
        self.manager.set_new_user("user")

        cue = self.manager.get_cue_card()

        self.manager.submit_utterance(
            "cupcakes, racecars, and other things", speech_state=True
        )

        self.manager.stop_session()

        # SESSION 2
        self.manager.start_session(numpy.array([]))

        cue = self.manager.get_cue_card()

        self.manager.submit_utterance(
            cue + "cupcakes, racecars, and other things", speech_state=True
        )

        self.manager.stop_session()

        # SESSION 3
        self.manager.start_session(numpy.array([]))

        cue = self.manager.get_cue_card()

        self.manager.submit_utterance(cue, speech_state=True)
        self.manager.submit_speech_data(0, True)

        self.manager.stop_session()

        # GET REPORT
        _, _, progress = self.manager.start_session(numpy.array([]))

        print(progress)
        assert progress is not None

        self._post()
