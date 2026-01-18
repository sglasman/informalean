def tokenize_for_interesting_words(s: str):
    tokens = set()
    buffer = []

    def empty_buffer():
        if buffer:
            tokens.add("".join(buffer))
        buffer.clear()

    for c in s:
        if c.isalnum() or c == "_":
            buffer.append(c)
        elif c.isspace() or c in ignored_chars:
            empty_buffer()
            continue
        else:
            empty_buffer()
            tokens.add(c)
    empty_buffer()
    return [token for token in tokens if is_interesting(token)]


def is_interesting(word):
    return word not in list(chars_and_nums)


def shingle(s: str, k: int) -> list[str]:
    return [s[i : i + k] for i in range(len(s) - k + 1)]


ignored_chars = "[](){}→:=+-.,"
chars_and_nums = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩᶠ"
open_brackets = "[({"
close_brackets = "])}"
declaration_words = {
    "def",
    "theorem",
    "lemma",
    "opaque",
    "inductive",
    "axiom",
    "example",
    "instance",
    "abbrev",
    "irreducible_def",
}
