from spacy.tokens.doc import Doc
from memory.processing.pipeline import Pipeline, PosPipe


class TestPosPipe:
    def test_pos_type(self):
        tagger = PosPipe()
        doc = tagger.process("This is a test sentence")

        assert isinstance(doc, list)

    def test_fluency(self):
        pipeline = Pipeline()
        tokens, metadata = pipeline.process("This is some text, and, again, some text")
        assert metadata.fluency_score != 9 and metadata.fluency_score != 0
