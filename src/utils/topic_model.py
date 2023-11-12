import pickle
from gensim.test.utils import common_dictionary
from spacy.tokens.doc import Doc
import re
import nltk
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim.utils import simple_preprocess


class TopicModel:
    def __init__(self, path: str = "assets/topic_model/LDA_model_32") -> None:
        # Downloads
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        self.model = self._load_model(path)
        # print(self.model)

    def _load_model(self, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    def get_topic_probability(
        self, tokens: list[str] | list[tuple[str, str]]
    ) -> dict[int, float]:
        """Returns a dictionary of topic probabilities indexed by their topic"""

        try:
            if isinstance(tokens[0], tuple):
                tokens = TopicModel.filter_tokens([token[0] for token in tokens])
        except IndexError:
            pass

        bag_of_words = common_dictionary.doc2bow(tokens)
        topics_prob_list: list[tuple[int, float]] = self.model.get_document_topics(
            bag_of_words
        )
        return dict(topics_prob_list)

    def get_topic_most_likely(
        self, tokens: list[str] | list[tuple[str, str]]
    ) -> tuple[int, float]:
        """Returns the most likely topic and its likelihood"""

        topic_probability = self.get_topic_probability(tokens)
        most_likely_topic: int = max(topic_probability, key=topic_probability.get)
        return most_likely_topic, topic_probability[most_likely_topic]

    def is_on_topic(
        self,
        tokens_a: list[str] | list[tuple[str, str]],
        tokens_b: list[str] | list[tuple[str, str]],
        threshold: float = 0.8,
    ) -> bool:
        """
        Returns a bool based on if the likelihood of tokens_b having a smilar
        probability for the highest topic of a
        """
        topic_a = self.get_topic_most_likely(tokens_a)
        topic_b_probabilities = self.get_topic_probability(tokens_b)
        # print(topic_a)
        # print(topic_b_probabilities)

        return topic_b_probabilities[topic_a[0]] / topic_a[1] > threshold

    @staticmethod
    def _remove_puctuation(document: str):
        return re.sub(r"[^a-zA-Z0-9]", " ", document)

    @staticmethod
    def _to_lower_case(document: str):
        return document.lower()

    @staticmethod
    def _remove_numbers(document: str):
        return "".join([i for i in document if not i.isdigit()])

    @staticmethod
    def _tokenize(document: str):
        return nltk.word_tokenize(document)

    @staticmethod
    def _remove_stopwords(document: list[str]):
        stop_words = nltk.corpus.stopwords.words("english")
        stop_words.extend(
            [
                "from",
                "subject",
                "re",
                "edu",
                "use",
                "like",
                "would",
                "one",
                "time",
                "make",
                "go",
                "also",
            ]
        )

        return [
            word for word in simple_preprocess(str(document)) if word not in stop_words
        ]

    @staticmethod
    def _make_biagrams(document: list[str]):
        bigram = gensim.models.Phrases(document, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod[document]

    @staticmethod
    def _lemmatization(document: list[str]):
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}

            return tag_dict.get(tag, wn.NOUN)

        texts_out = []

        lemma_function = WordNetLemmatizer()
        for token in document:
            lemma = lemma_function.lemmatize(token, get_wordnet_pos(token))
            texts_out.append(lemma)

        return texts_out

    @staticmethod
    def cleanup_and_tokenize(text: str) -> list[str]:
        text = TopicModel._remove_puctuation(text)
        text = TopicModel._to_lower_case(text)
        text = TopicModel._remove_numbers(text)
        return TopicModel._tokenize(text)

    @staticmethod
    def filter_tokens(tokens: list[str]) -> list[str]:
        tokens = TopicModel._remove_stopwords(tokens)
        tokens = TopicModel._make_biagrams(tokens)  # TODO Check type?
        tokens = TopicModel._lemmatization(tokens)
        return tokens

    @staticmethod
    def preprocess(text: str | list[str]) -> list[str]:

        if isinstance(text, str):
            text = TopicModel.cleanup_and_tokenize(text)

        return TopicModel.filter_tokens(text)
