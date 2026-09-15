# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import math
import random

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def _features(self, sentence):
        return [word.lower() for word in sentence]

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for feature in self._features(sentence):
            index = self.indexer.add_and_get_index(feature, add=add_to_indexer)
            if index != -1:
                features[index] += 1
        return features


class BigramFeatureExtractor(UnigramFeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def _features(self, sentence):
        words = [word.lower() for word in sentence]
        # Each adjacent pair is an indicator, regardless of repetition.
        return dict.fromkeys(zip(words, words[1:]))


class BetterFeatureExtractor(UnigramFeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def _features(self, sentence):
        # Keep feature families distinct, including negation-scoped words.
        words = [word.lower() for word in sentence]
        features = [("unigram", word) for word in words]
        features.extend(("bigram", left, right) for left, right in zip(words, words[1:]))
        negated = False
        for word in words:
            if word in {".", ",", "!", "?", ";", ":"}:
                negated = False
            elif word in {"not", "no", "never", "n't"} or word.endswith("n't"):
                negated = True
            elif negated:
                features.append(("negated", word))
        # Binary presence reduces the influence of repeated words.
        return dict.fromkeys(features)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor: FeatureExtractor, bias: float=0.0):
        self.weights = dict(weights)
        self.feat_extractor = feat_extractor
        self.bias = bias

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = self.bias + sum(self.weights.get(index, 0.0) * value
                                for index, value in features.items())
        return int(score >= 0.0)


class LogisticRegressionClassifier(PerceptronClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    # A sigmoid probability is >= 0.5 exactly when its linear score is >= 0.
    # The constructor and binary prediction are inherited from PerceptronClassifier.



def _training_features(train_exs, feat_extractor):
    if not train_exs:
        raise ValueError("Training requires at least one example")
    examples = []
    for example in train_exs:
        if example.label not in (0, 1):
            raise ValueError("Sentiment labels must be 0 or 1")
        features = feat_extractor.extract_features(example.words, add_to_indexer=True)
        # Indexer assigns nonnegative indices; -1 is reserved for the intercept.
        features[-1] = 1
        examples.append((features, example.label))
    return examples


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    examples = _training_features(train_exs, feat_extractor)
    weights = Counter()
    totals = Counter()
    timestamps = Counter()
    rng = random.Random(0)
    step = 0
    for epoch in range(20):
        rng.shuffle(examples)
        for features, label in examples:
            score = sum(weights[index] * value for index, value in features.items())
            error = label - int(score >= 0.0)
            if error:
                for index, value in features.items():
                    # Lazy averaging counts unchanged weights between updates.
                    totals[index] += (step - timestamps[index]) * weights[index]
                    timestamps[index] = step
                    weights[index] += error * value
            step += 1
    averaged = {index: (totals[index] + (step - timestamps[index]) * weight) / step
                for index, weight in weights.items()}
    bias = averaged.pop(-1, 0.0)
    return PerceptronClassifier(averaged, feat_extractor, bias)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    examples = _training_features(train_exs, feat_extractor)
    weights = Counter()
    rng = random.Random(0)
    for epoch in range(25):
        rng.shuffle(examples)
        learning_rate = 0.15 / (1.0 + 0.1 * epoch)
        for features, label in examples:
            score = sum(weights[index] * value for index, value in features.items())
            # This form avoids overflow for either sign of the score.
            if score >= 0.0:
                probability = 1.0 / (1.0 + math.exp(-score))
            else:
                exp_score = math.exp(score)
                probability = exp_score / (1.0 + exp_score)
            update = learning_rate * (label - probability)
            for index, value in features.items():
                weights[index] += update * value
    bias = weights.pop(-1, 0.0)
    return LogisticRegressionClassifier(weights, feat_extractor, bias)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
