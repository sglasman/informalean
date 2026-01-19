CHINESE_RANGES = [
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0x2F800, 0x2FA1F),  # Compatibility Ideographs Supplement
]


def is_chinese_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in CHINESE_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def chinese_char_fraction(text: str) -> float:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if len(alpha_chars) == 0:
        return 0
    return len([ch for ch in alpha_chars if is_chinese_char(ch)]) / len(alpha_chars)
