"""
Voting strategies for multi-agent consensus.

This module contains different voting methods that can be used across
all topologies to determine the final group answer.

To change the voting method, modify the VOTING_METHOD constant or
call a different function directly.
"""

from typing import Optional, Union


def majority_vote(
    answers: list[Union[float, int, None]],
) -> Optional[Union[float, int]]:
    """
    Pick the most common answer. If tied, pick first.
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None

    # Count occurrences
    counts = {}
    for a in valid_answers:
        counts[a] = counts.get(a, 0) + 1

    # Pick answer with most votes (first if multiple share max)
    return max(counts, key=counts.get)


def confidence_weighted_vote(responses: list[dict]) -> Optional[Union[float, int]]:
    """
    Weight answers by confidence and pick the highest.
    """
    valid_responses = [r for r in responses if r.get("answer") is not None]

    if not valid_responses:
        return None

    # Pick response with highest confidence (first if tied)
    return max(valid_responses, key=lambda r: r.get("confidence", 0))["answer"]


def median_vote(answers: list[Union[float, int, None]]) -> Optional[Union[float, int]]:
    """
    Pick the median answer (middle value when sorted).
    """
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None

    sorted_answers = sorted(valid_answers)
    n = len(sorted_answers)

    if n % 2 == 1:
        return sorted_answers[n // 2]
    else:
        # For even count, return the lower of the two middle values
        return sorted_answers[n // 2 - 1]


# Default voting method - change this to switch strategies
# Options: majority_vote, confidence_weighted_vote, median_vote
VOTING_METHOD = majority_vote
