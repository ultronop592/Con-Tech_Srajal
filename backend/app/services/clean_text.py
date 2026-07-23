import re


COMMON_OCR_REPLACEMENTS = [
    (re.compile(r"\bTeant\b", re.IGNORECASE), "Tenant"),
    (re.compile(r"\bLicensorr\b", re.IGNORECASE), "Licensor"),
    (re.compile(r"\bLicenseee\b", re.IGNORECASE), "Licensee"),
    (re.compile(r"\bLessorr\b", re.IGNORECASE), "Lessor"),
    (re.compile(r"\bLesee\b", re.IGNORECASE), "Lessee"),
    (re.compile(r"\bRs\.?\s*10O00\b", re.IGNORECASE), "Rs. 10,000"),
    (re.compile(r"\bper\s+annum\b", re.IGNORECASE), "per year"),
]


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)

    # Apply legal OCR repair rules
    for pattern, replacement in COMMON_OCR_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)

    return normalized.strip()

