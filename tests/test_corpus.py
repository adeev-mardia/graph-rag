from graphrag.corpus import split_sentences


def test_basic_split():
    text = "The sky is blue. The grass is green."
    assert split_sentences(text) == ["The sky is blue.", "The grass is green."]


def test_middle_initial_not_split():
    text = "President John F. Kennedy gave a speech. It was short."
    sentences = split_sentences(text)
    assert sentences[0] == "President John F. Kennedy gave a speech."
    assert sentences[1] == "It was short."


def test_abbreviation_not_split():
    text = "Dr. Smith led the team. The team succeeded."
    sentences = split_sentences(text)
    assert sentences[0] == "Dr. Smith led the team."
    assert len(sentences) == 2


def test_number_followed_by_letter_not_split():
    text = "It happened in the 1990s. Things changed after that."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "It happened in the 1990s."
