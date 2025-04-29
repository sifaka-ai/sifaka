import warnings
import sys

# Enable all warnings
warnings.filterwarnings("always")

# Import modules that might trigger warnings
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.spam import SpamClassifier
from sifaka.classifiers.toxicity import ToxicityClassifier

# Create instances to trigger any initialization warnings
print("Creating classifier instances...")
language_classifier = LanguageClassifier()
profanity_classifier = ProfanityClassifier()
readability_classifier = ReadabilityClassifier()
sentiment_classifier = SentimentClassifier()
spam_classifier = SpamClassifier()
toxicity_classifier = ToxicityClassifier()

print("Done.")
