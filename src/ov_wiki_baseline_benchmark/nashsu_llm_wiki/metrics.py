"""Metrics copied from the reference OV baseline's benchmark/wiki implementation."""

from __future__ import annotations

import collections
import re
import string


def normalize_answer(value: str) -> str:
    text = str(value).replace(",", "")
    text = re.sub(r"\b(a|an|the|and)\b", " ", text.lower())
    punctuation = set(string.punctuation)
    text = "".join(character for character in text if character not in punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    predicted = normalize_answer(prediction).split()
    truth = normalize_answer(ground_truth).split()
    common = collections.Counter(predicted) & collections.Counter(truth)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    return (2 * precision * recall) / (precision + recall)


def max_token_f1(prediction: str, gold_answers: list[str]) -> float:
    return max((token_f1(prediction, gold) for gold in gold_answers), default=0.0)


def is_refusal(text: str) -> bool:
    phrases = [
        "not mentioned",
        "no information",
        "cannot be answered",
        "none",
        "unknown",
        "don't know",
    ]
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)
