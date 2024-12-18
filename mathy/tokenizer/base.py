from collections import Counter
from typing import Dict, List

class Document:
    """
    A document is a sequence of single character tokens including special tokens 
    such as start of text, end of text, and end of word.  It acts as a iterator over
    the tokens in the document.
    """

    start_of_text_token: str = "<s>"
    end_of_text_token: str = "</s>"
    end_of_word_token: str = "</w>"

    def __iter__(self):
        raise NotImplementedError("Subclasses must implement the _generate_tokens method.")
    
    def __next__(self):
        raise NotImplementedError("Subclasses must implement the __next__ method.")


class StringDocument(Document):
    text: str

    def __init__(self, text: str):
        self.text = text

    def __iter__(self):
        self.next = self._generate_tokens()
        return self

    def _generate_tokens(self):
        words = self.text.split()
        yield self.start_of_text_token
        for word in words:
            for char in word:
                yield char
            yield self.end_of_word_token
        yield self.end_of_text_token

    def __next__(self):
        return next(self.next)

class Tokenizer:
    """
    Base class for tokenizers.
    """

    def train(self, corpus: List[str]):
        """
        Train the tokenizer on a corpus.
        """
        raise NotImplementedError("Subclasses must implement the train method.")

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize a string into a list of integers.
        """
        raise NotImplementedError("Subclasses must implement the tokenize method.")

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of integers into a string.
        """
        raise NotImplementedError("Subclasses must implement the decode method.")
    

class CharacterTokenizer(Tokenizer):
    """
    Tokenizer that splits text into characters.  The tokenizer is basically a BPE tokenizer with
    no merges.  This means that the vocabulary is just the set of characters in the corpus plus
    the start of text, end of text, and end of word tokens.
    """

    vocab: Counter
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    padding_token_id: int
    padding_token: str = "<pad>"

    start_of_text_token_id: int
    start_of_text_token: str = Document.start_of_text_token

    end_of_text_token_id: int
    end_of_text_token: str = Document.end_of_text_token

    end_of_word_token_id: int
    end_of_word_token: str = Document.end_of_word_token

    def add_new_token(self, token: str):
        self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token[self.token_to_id[token]] = token

    def __init__(self):
        self.vocab = Counter()
        self.token_to_id = {}
        self.id_to_token = {}

        self.add_new_token(self.padding_token)
        self.add_new_token(self.start_of_text_token)
        self.add_new_token(self.end_of_text_token)
        self.add_new_token(self.end_of_word_token)

        self.padding_token_id = self.token_to_id[self.padding_token]
        self.start_of_text_token_id = self.token_to_id[self.start_of_text_token]
        self.end_of_text_token_id = self.token_to_id[self.end_of_text_token]
        self.end_of_word_token_id = self.token_to_id[self.end_of_word_token]


    def train(self, corpus: List[str]):
        documents = [StringDocument(sentence) for sentence in corpus]
        for document in documents:
            for token in document:
                self.vocab[token] += 1
                if token not in self.token_to_id:
                    self.add_new_token(token)

    def tokenize(self, text: str, context_length: int) -> List[int]:
        document = StringDocument(text)
        tokens = [self.token_to_id[token] for token in document]
        # truncate if necessary
        tokens = tokens[:context_length]
        # pad if necessary
        return tokens + [self.padding_token_id] * (context_length - len(tokens))
    
    def decode(self, tokens: List[int]) -> str:
        if self.padding_token_id in tokens:
            tokens = tokens[:tokens.index(self.padding_token_id)]
        return "".join([self.id_to_token[token] for token in tokens])
    
    def vocab_size(self) -> int:
        return len(self.token_to_id)
