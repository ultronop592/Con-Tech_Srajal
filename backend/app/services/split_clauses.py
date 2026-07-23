import re


def split_into_clauses(text: str, max_clause_chars: int = 600) -> list[str]:
    if not text:
        return []

    # Enhanced regex to capture digits (1., 1.1), clause/section/article headers,
    # Roman numerals (I., II., III.), lettered lists (A., a.), and numbered parentheses ((1), (a))
    split_pattern = r"\n(?=(?:\d+(?:\.\d+)*\.|clause\s+\d+|section\s+\d+|article\s+\d+|paragraph\s+\d+|(?:M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.|[A-Z]\.|\([0-9a-z]\)|-\s))"
    chunks = [
        chunk.strip()
        for chunk in re.split(split_pattern, text, flags=re.IGNORECASE)
        if chunk.strip()
    ]

    clauses: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_clause_chars:
            clauses.append(chunk)
            continue

        sentence_parts = re.split(r"(?<=[.!?])\s+", chunk)
        buffer = ""
        for sentence in sentence_parts:
            candidate = f"{buffer} {sentence}".strip()
            if len(candidate) <= max_clause_chars:
                buffer = candidate
            else:
                if buffer:
                    clauses.append(buffer)
                buffer = sentence.strip()
        if buffer:
            clauses.append(buffer)

    return clauses
