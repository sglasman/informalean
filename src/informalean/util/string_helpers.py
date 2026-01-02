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


ignored_chars = "[](){}→:=+-.,"
chars_and_nums = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩᶠ"
