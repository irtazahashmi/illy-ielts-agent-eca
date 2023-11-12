from utils.topic_model import TopicModel


class TestTopicModel:
    def test_simple(self):
        model = TopicModel()
        assert model.is_on_topic(["one", "topic"], ["one", "topic"])
