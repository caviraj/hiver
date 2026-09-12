"""Contraction mapping dictionary for lexical de-contraction.

Contains standard English contractions and social media shorthand commonly found
in customer support conversations on platforms like Twitter/X.
"""

from typing import Dict

# NOTE on Ambiguous Contractions:
# Certain contractions in English are ambiguous in written form without deeper POS or
# syntactic analysis:
# - "it's" -> "it is" vs "it has"
# - "he's" -> "he is" vs "he has"
# - "she's" -> "she is" vs "she has"
# - "that's" -> "that is" vs "that has"
# - "there's" -> "there is" vs "there has"
# - "i'd" -> "i would" vs "i had"
#
# In customer support tweets (e.g. "@AppleSupport my phone is dead, it's not charging"),
# "it's" is used as "it is" in over 95% of occurrences. Per PRD guidelines, we default
# to the high-probability expansion ("it is", "i would", etc.). This is a known,
# deliberate heuristic to prioritize downstream embeddings and entity normalization.

CONTRACTIONS: Dict[str, str] = {
    # Negations
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "cannot": "cannot",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    # Pronoun + 'm / 're / 's / 've / 'll / 'd
    "i'm": "i am",
    "i'll": "i will",
    "i'd": "i would",
    "i've": "i have",
    "you're": "you are",
    "you'll": "you will",
    "you'd": "you would",
    "you've": "you have",
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "it's": "it is",
    "it'll": "it will",
    "it'd": "it would",
    "we're": "we are",
    "we'll": "we will",
    "we'd": "we would",
    "we've": "we have",
    "they're": "they are",
    "they'll": "they will",
    "they'd": "they would",
    "they've": "they have",
    # Demonstratives and Wh- words
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "there's": "there is",
    "there'll": "there will",
    "there'd": "there would",
    "here's": "here is",
    "what's": "what is",
    "what're": "what are",
    "what'll": "what will",
    "what've": "what have",
    "where's": "where is",
    "where'd": "where did",
    "where'll": "where will",
    "who's": "who is",
    "who'll": "who will",
    "who'd": "who would",
    "who've": "who have",
    "how's": "how is",
    "how'd": "how did",
    "how'll": "how will",
    "when's": "when is",
    "why's": "why is",
    # Modal + have
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    # Miscellaneous Standard
    "let's": "let us",
    "o'clock": "of the clock",
    "ma'am": "madam",
    # Social Media & Informal Shorthand (PRD Requirements)
    "'bout": "about",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "dunno": "do not know",
    "y'all": "you all",
    "kinda": "kind of",
    "sorta": "sort of",
    "imma": "i am going to",
    "'cause": "because",
}
