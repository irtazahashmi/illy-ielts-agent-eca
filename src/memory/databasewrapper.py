from __future__ import annotations

import pickle
import time
import face_recognition
import numpy

from numpy._typing import NDArray
import pandas as pd

from pymongo import MongoClient
from pymongo.cursor import Cursor
from spacy.tokens.doc import Doc

from utils.topic_model import TopicModel


class Storable:
    """Abstract object"""

    def _to_mongo_obj(self) -> dict:
        """Converts the object to a dict that is encodable by mongodb"""
        raise NotImplementedError("Storable object doesn't implement _to_mongo method")

    @staticmethod
    def _from_mongo_obj(obj: dict):
        """Converts a mongodb object to the class"""
        raise NotImplementedError(
            "Storable object doesn't implement _from_mongo method"
        )

    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError(
            "Storable object doesn't implement equals method (necessary for db retrieval)"
        )

    def __str__(self) -> str:
        raise NotImplementedError(
            "Storable object doesn't implement string method (necessary for hashing)"
        )

    def __hash__(self) -> int:
        return hash(str(self))


class User(Storable):
    """Object containing user data [long-term memory]"""

    def __init__(self, name: str, encodings: NDArray, _id: int | None = None) -> None:
        self.name: str = name
        self.face_encodings: NDArray = encodings

        self._id: int = _id if _id is not None else hash(self)

    def _to_mongo_obj(self) -> dict:
        return {
            "_id": self._id,
            "name": self.name,
            "face_encodings": pickle.dumps(self.face_encodings),
        }

    @staticmethod
    def _from_mongo_obj(obj: dict) -> "User":
        return User(obj["name"], pickle.loads(obj["face_encodings"]), _id=obj["_id"])

    def __str__(self) -> str:
        return self.name + str(self.face_encodings)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, User)
            and self.name == __o.name
            and numpy.array_equal(self.face_encodings, __o.face_encodings)
        )

    def __hash__(self) -> int:
        return hash(str(self))


class Session(Storable):
    """Object containing session data [long-term memory]"""

    def __init__(
        self,
        user: User,
        _id: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        cue_card_id: int | None = None,
        follow_ups_idx: list[int] = [],
        average_score: float = 0.0,
        on_topic: float = 0.0,
        over_time: bool = False,
    ) -> None:
        self.user = user
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time = end_time
        self.cue_card_id = cue_card_id
        self.follow_ups_idx = follow_ups_idx
        self.average_score = average_score
        self.on_topic = on_topic
        self.over_time = over_time

        self._id = _id if _id is not None else hash(self)

    def _to_mongo_obj(self) -> dict:
        return {
            "_id": self._id,
            "user": self.user._to_mongo_obj(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "cue_card_id": self.cue_card_id,
            "follow_ups_idx": self.follow_ups_idx,
            "average_score": self.average_score,
            "on_topic": self.on_topic,
            "over_time": self.over_time,
        }

    @staticmethod
    def _from_mongo_obj(obj: dict) -> "Session":
        return Session(
            _id=obj["_id"],
            user=User._from_mongo_obj(obj["user"]),
            start_time=obj["start_time"],
            end_time=obj["end_time"],
            cue_card_id=obj["cue_card_id"],
            follow_ups_idx=obj["follow_ups_idx"],
            average_score=obj["average_score"],
            on_topic=obj["on_topic"],
            over_time=obj["over_time"],
        )

    def __str__(self) -> str:
        return f"[\
            {self.start_time} : \
            {self.end_time if self.end_time is not None else -1}\
        ] {self.user}"

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, Session)
            and self.start_time == __o.start_time
            and self.user == __o.user
        )

    def __hash__(self) -> int:
        return hash(str(self))


class MetaData(Storable):
    """
    Storable that accompanies tokens containing any relevent data resulting
    from processing or otherwise
    """

    def __init__(self, topic: int, fluency_score: int, _id: int | None = None) -> None:
        self.topic: int = topic
        self.fluency_score: int = fluency_score
        self._id: int = hash(self) if _id is None else _id

    def _to_mongo_obj(self) -> dict:
        return {
            "topic": self.topic,
            "fluency_score": self.fluency_score,
        }

    @staticmethod
    def _from_mongo_obj(obj: dict):
        return MetaData(obj["topic"], obj["fluency_score"])

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, MetaData)
            and self.topic == __o.topic
            and self.fluency_score == __o.fluency_score
        )

    def __str__(self) -> str:
        return f"[topic:{self.topic};fluency_score:{self.fluency_score}]"

    def __hash__(self) -> int:
        return hash(str(self))


class Utterance(Storable):
    def __init__(
        self,
        tokens: list[tuple[str, str]],
        timestamp: float,
        speech_state: bool,
        metadata: MetaData | None = None,
        _id: int | None = None,
    ) -> None:
        self.tokens: list[tuple[str, str]] = tokens
        self.timestamp: float = timestamp
        self.speech_state: bool = speech_state
        self.metadata: MetaData | None = metadata
        self._id = hash(self) if _id is None else _id

    def _to_mongo_obj(self) -> dict:
        return {
            "tokens": pickle.dumps(self.tokens),
            "timestamp": self.timestamp,
            "speech_state": self.speech_state,
            "metadata": self.metadata
            if self.metadata is None
            else self.metadata._to_mongo_obj(),
        }

    @staticmethod
    def _from_mongo_obj(obj: dict) -> "Utterance":
        metadata = None

        try:
            metadata = obj["metadata"]
        except KeyError:
            pass

        return Utterance(
            tokens=pickle.loads(obj["tokens"]),
            timestamp=obj["timestamp"],
            speech_state=obj["speech_state"],
            metadata=metadata
            if metadata is None
            else MetaData._from_mongo_obj(metadata),
        )

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Utterance) and str(self) == str(__o)

    def __str__(self) -> str:
        """Return Document to string that was submitted"""
        sentence = "".join([token[0] for token in self.tokens])

        return f"[{self.metadata}] : {sentence}"

    def __hash__(self) -> int:
        return hash(str(self))


class Database:
    """Database object which abstracts the different interactions between types of databases"""

    CONNECTION_STRING = "mongodb://localhost/myFirstDatabase"
    questionbank = "assets/question_bank.csv"

    def __init__(self, db_name: str | None = None, clear: bool = False):
        if db_name is None:
            self._connect_database(clear=clear)
        else:
            self._connect_database(db_name=db_name, clear=clear)

    def get_cue_card_random(self) -> tuple[str, int]:
        """
        Returns a random cue card along with its _id
        TODO intelligently pick cue cards
        """
        reader = pd.read_csv(self.questionbank)
        sample = reader.sample()
        row_number = sample.index[0]
        return (sample["topic"][row_number], int(row_number))

    def get_cue_card_by_id(self, _id: int) -> str:
        """Returns a cue card based on the given _id"""

        reader = pd.read_csv(self.questionbank)
        return reader["topic"][_id]

    def get_follow_up_by_id(self, _id: int) -> list[str]:
        """
        Returns follow-up questions corresponding to the cue card whose index is given
        TODO intelligently pick follow-up question
        """
        reader = pd.read_csv(self.questionbank)
        return reader["questions"][_id].split("|")

    def user_from_encodings(self, face_encodings: NDArray) -> User | None:
        for other in self._get_all_users():
            compared_enc = face_recognition.compare_faces(
                [face_encodings], other.face_encodings
            )
            # print(compared_enc)
            if compared_enc[0]:
                return other

        return None

    def flush_short_term(self, session: Session):
        """
        Takes the current short-term memory and stores (the required parts) into long-term memory
        TODO Actually develop a type of short-term memory
        """
        assert session.cue_card_id is not None

        utterances: list[Utterance] = self._get_all_utterances()
        self.utterances.drop()
        scores = []

        for utterance in utterances:
            if utterance.metadata is not None:
                scores.append(utterance.metadata.fluency_score)

        session.average_score = sum(scores) / len(scores) if len(scores) != 0.0 else 0.0

        self._insert_user(session.user)
        self._insert_session(session)

    def insert_utterance(self, utterance: Utterance):
        """Inserts given utterance into the database"""
        self.utterances.insert_one(utterance._to_mongo_obj())

    def get_last_utterance(self) -> Utterance | None:
        """Inserts"""
        objs = self.utterances.aggregate(
            [
                {"$group": {"_id": "$timestamp", "docs": {"$push": "$$ROOT"}}},
                {"$sort": {"_id": 1}},
                {"$limit": 1},
            ]
        )

        # Mongo returns a Cursor object which needs to be iterated over
        for obj in objs:
            return Utterance._from_mongo_obj(obj["docs"][0])

    def _get_all_utterances(self) -> list[Utterance]:
        objs = self.utterances.find()
        return [Utterance._from_mongo_obj(obj) for obj in objs]

    # USERS
    def _insert_user(self, user: User) -> None:
        obj = self.users.find_one({"_id": user._id})

        if obj is None:
            self.users.insert_one(user._to_mongo_obj())

    def _get_user_by_name(self, user_name: str) -> User | None:
        obj = self.users.find_one({"name": user_name})

        return User._from_mongo_obj(obj) if obj is not None else obj

    def _get_all_users(self) -> list[User]:
        objs: Cursor = self.users.find()
        return [User._from_mongo_obj(obj) for obj in objs]

    # SESSIONS
    def _insert_session(self, session: Session) -> None:
        self.sessions.insert_one(session._to_mongo_obj())

    def get_sessions_by_user(self, user: User) -> list[Session]:
        objs: Cursor = self.sessions.find({"user._id": user._id})
        return [Session._from_mongo_obj(obj) for obj in objs]

    def _get_all_sessions(self) -> list[Session]:
        objs: Cursor = self.sessions.find()
        return [Session._from_mongo_obj(obj) for obj in objs]

    # DATABASE
    def _connect_database(
        self, db_name: str = "agent_illy", clear: bool = False
    ) -> None:
        """
        Connects database and initializes required collections.
        Clears the long-term memory collections (currently "users" and "sessions") if clear is set
        """

        self._client = MongoClient(self.CONNECTION_STRING)
        self._db = self._client[db_name]

        if clear:
            self._drop_db(db_name)
            self._client = MongoClient(self.CONNECTION_STRING)[db_name]

        # long-term
        self.users = self._db["users"]
        self.sessions = self._db["sessions"]
        self.users.create_index("name")
        self.sessions.create_index("user")

        # short-term
        self.utterances = self._db["utterances"]
        self.utterances.create_index("timestamp")

    def _drop_db(self, db_name: str) -> None:
        """Deletes the database by the given name (used for testing purposes)"""
        assert self._client is not None
        self._client.drop_database(db_name)
