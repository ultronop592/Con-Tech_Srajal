import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Regular expressions for numeric, monetary, and temporal entities
NUMERIC_PATTERN = re.compile(r"\b\d+(?:,\d+)*(?:\.\d+)?\b")
MONEY_PATTERN = re.compile(r"\b(?:rs\.?|inr|rupees?)\s*\d+(?:,\d+)*\b|\b\d+(?:,\d+)*\s*(?:rs\.?|inr|rupees?)\b", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"\b\d+\s*(?:days?|months?|years?|weeks?|hours?)\b", re.IGNORECASE)


def extract_entities(text: str) -> dict:
    """Extract numbers, monetary amounts, and time durations from text."""
    lowered = text.lower()
    numbers = set(NUMERIC_PATTERN.findall(lowered))
    money = set(MONEY_PATTERN.findall(lowered))
    durations = set(DURATION_PATTERN.findall(lowered))
    return {
        "numbers": numbers,
        "money": money,
        "durations": durations,
    }


def verify_and_clean_grounding(clause: str, plain_english: str) -> Tuple[str, bool]:
    """
    Verifies that numeric, monetary, and temporal facts in generated plain English
    are grounded in the original source clause.

    Returns:
        Tuple of (cleaned_text, sentence_was_modified)
    """
    if not plain_english or not clause:
        return plain_english, False

    source_lowered = clause.lower()
    target_lowered = plain_english.lower()

    target_numbers = set(NUMERIC_PATTERN.findall(target_lowered))
    source_numbers = set(NUMERIC_PATTERN.findall(source_lowered))

    # Check for ungrounded numbers (e.g. LLM invented '10' or '5000' when source has none)
    ungrounded_numbers = target_numbers - source_numbers

    if ungrounded_numbers:
        logger.warning(
            "Hallucination shield triggered: target text contains ungrounded numbers %s not found in source clause.",
            ungrounded_numbers,
        )
        cleaned = plain_english
        for num in ungrounded_numbers:
            # If the number is part of a phrase like "Rs. 5000", strip or replace the whole phrase
            pattern = re.compile(rf"\b(?:rs\.?|inr|rupees?)?\s*{re.escape(num)}\s*(?:rupees|days|months|years)?\b", re.IGNORECASE)
            cleaned = pattern.sub("the specified amount/period", cleaned)
        
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, True

    return plain_english, False
