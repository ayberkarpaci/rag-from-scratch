from src.preprocessing import clean_text


def test_adds_space_after_sentence_end():
    assert clean_text("First answer.Second answer") == "First answer. Second answer"


def test_separates_closing_quote_from_next_word():
    assert clean_text('ends with 5"and more') == 'ends with 5" and more'


def test_collapses_doubled_quotes():
    assert clean_text('a "" b') == 'a " b'


def test_collapses_repeated_spaces_and_strips():
    assert clean_text("  one   two  ") == "one two"
