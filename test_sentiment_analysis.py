from SentimentAnalysis.sentiment_analysis import sentiment_analyzer
import unittest


class TestSentimentAnalyzer(unittest.TestCase):
    def test_sentiment_analyzer(self):
        self.assertEqual(self.get_label("I love working with Python"),
                         "SENT_POSITIVE")
        self.assertEqual(self.get_label("I hate working with Pyhton"),
                         "SENT_NEGATIVE")
        self.assertEqual(self.get_label("I am neutral on Python"),
                         "SENT_NEUTRAL")

    def get_label(self, text):
        return sentiment_analyzer(text)['label']


if __name__ == "__main__":
    unittest.main()
