from collections import Counter
from spacy.tokens.doc import Doc

from utils.topic_model import TopicModel


class LanguageFluency:
    """Given a user input (text), determine user IELTS speaking fluency"""

    def _count_repetitions(self, tokens: list[tuple[str, str]]) -> Counter[str]:
        words = TopicModel.filter_tokens([token[0] for token in tokens])
        return Counter(words)

    def _calculate_fluency_score(self, repetitions: Counter[str]) -> int:
        """Calculate the speaking fluency"""

        most_frequent_number_of_repetitions = 3

        # TODO: Check the rules for assinging the score (1 - 9)
        if len(repetitions.most_common(None)) != 0:
            most_frequent_number_of_repetitions = repetitions.most_common(None)[0][1]

        # TODO: Check the rules for assinging the score (1 - 9)
        if most_frequent_number_of_repetitions == 1:
            return 9
        if most_frequent_number_of_repetitions == 2:
            return 8
        if most_frequent_number_of_repetitions == 3:
            return 7
        if most_frequent_number_of_repetitions == 3:
            return 6
        if most_frequent_number_of_repetitions == 3:
            return 5
        if most_frequent_number_of_repetitions == 3:
            return 4
        if most_frequent_number_of_repetitions == 3:
            return 3
        return 2

    def _fluency_category(self, fluency_score: int) -> str:
        """
        Return the speaking fluency rubric
        Test the following w.r.t. the given topic:
        * Use of connectives
        * Range of vocabulary
        """

        # First category (7-9) -> advanced
        # Not many repetitions, no hesitation, no pauses, no false starts
        if fluency_score >= 7:
            language_category = "Advanced"
        # Second category (4-6) -> intermediate
        # repetitious use of simple connectives
        elif fluency_score >= 4:
            language_category = "Intermediate"
        # Third category (1-3) -> beginner
        # Only simple message
        elif fluency_score >= 1:
            language_category = "Beginner"
        else:
            language_category = "Unqualified"

        return language_category

    def get_fluency(self, tokens: list[tuple[str, str]]) -> tuple[int, str]:
        """Return the speaking fluency in terms of score and category"""
        repetitions = self._count_repetitions(tokens)
        score = self._calculate_fluency_score(repetitions)
        return (score, self._fluency_category(score))
