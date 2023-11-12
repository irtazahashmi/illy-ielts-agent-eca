from __future__ import annotations
import face_recognition
from numpy._typing import NDArray
import pandas as pd
import time


class User:
    """Object containing user data [long-term memory]"""

    name: str
    face_encodings: NDArray
    sessions: list["Session"] = []  # TODO Should refer to entries in database

    def __init__(self, name: str, encodings: NDArray) -> None:
        self.name = name
        self.face_encodings = encodings

    def __str__(self) -> str:
        return self.name + str(self.face_encodings)


class Session:
    """Object containing session data [long-term memory]"""

    user: User  # TODO Should refer to entry in database
    id: int

    start_time: float
    end_time: float

    active: bool = True

    cue_card_id: int | None = None
    follow_ups_idx: list[int] = []

    def __init__(self, user: User) -> None:
        self.user = user
        self.start_time = time.time()
        self.id = hash(str(self.user) + str(self.start_time))


class Database:
    # TODO Make use of actual (persistent) database
    questionbank = "assets/question_bank.csv"
    users: list[User] = []  # HACK Just for testing

    def get_cue_card_random(self) -> tuple[str, int]:
        reader = pd.read_csv(self.questionbank)
        sample = reader.sample()
        row_number = sample.index[0]
        return (sample["topic"][row_number], row_number)

    def get_cue_card_by_id(self, id: int) -> str:
        reader = pd.read_csv(self.questionbank)
        return reader["topic"][id]

    def get_follow_up_by_id(self, id: int) -> list[str]:
        reader = pd.read_csv(self.questionbank)
        return reader["questions"][id].split("|")

    def user_from_encodings(self, face_encodings: NDArray) -> User | None:
        for other in self.users:
            compared_enc = face_recognition.compare_faces(
                [face_encodings], other.face_encodings
            )
            print(compared_enc)
            if compared_enc[0]:
                return other

        return None

    def flush_short_term(self, user: User):
        self.users.append(user)