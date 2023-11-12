from transformers import pipeline


class AffectModel():
    """
    Current dialog state
    """
    def __init__(self):
        self.model = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

    def predict(self, text: str) -> str:
        """ returns the emotion from text """
        emotion = self.model(text)[0]['label']
        #print(f"predicted affect was: {emotion}")
        return emotion
