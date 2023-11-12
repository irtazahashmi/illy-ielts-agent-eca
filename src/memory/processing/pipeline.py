import nltk
import spacy
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
from memory.databasewrapper import MetaData

from memory.processing.fluency import LanguageFluency
from utils.topic_model import TopicModel

# TODO Should we consider a processing window size, or should this be determined
# by the amount of information given? In the report we mentioned we wanted to
# employ incremental processing.

# TODO This whole thing is probably overkill, but it does help with separating
# some of the complexites of integrating the different processing steps later,
# so will leave it for now


class Pipe:
    """Abstract class which processing text and returns metadata"""

    def __init__(self) -> None:
        pass

    def process(self, data):
        """Processes the given data to generate some metadata"""
        pass


class PosPipe(Pipe):
    """Performs POS tagging on given text input"""

    def process(self, text: str) -> list[tuple[str, str]]:
        return nltk.pos_tag(TopicModel.cleanup_and_tokenize(text))


class TopicPipe(Pipe):
    model = TopicModel()

    def process(self, tokens: list[tuple[str, str]]):
        untagged_tokens = [token[0] for token in tokens]
        self.model.get_topic_probability(TopicModel.preprocess(untagged_tokens))


class FluencyPipe(Pipe):
    fluency_scorer = LanguageFluency()

    def process(self, tokens: list[tuple[str, str]]) -> int:
        return self.fluency_scorer.get_fluency(tokens)[0]


class Pipeline:
    """Object that will do processing of a single utterance and return meta-data"""

    def __init__(self):
        self.pipes: list[Pipe] = [PosPipe(), TopicPipe(), FluencyPipe()]

    def process(self, text: str) -> tuple[list[tuple[str, str]], MetaData]:
        """
        Processes the given text according to the constructed pipeline
        TODO Automatically construct dependency graph based on dependencies
        """
        tokens = self.pipes[0].process(text)
        #print(tokens)
        current_topic = self.pipes[1].process(tokens)
        fluency_score = self.pipes[2].process(tokens)

        return tokens, MetaData(current_topic, fluency_score)
