# tests for tokenizer
import unittest
from mathy.tokenizer import CharacterTokenizer

class TokenizerTest(unittest.TestCase):

    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_character_tokenizer(self):
        corpus = ["This is an example sentence", "Byte Pair Encoding is interesting", "Let's tokenize text"]
        self.tokenizer.train(corpus)

        input_text = "This is an example sentence"
        expected_text = "<s>This</w>is</w>an</w>example</w>sentence</w></s>" # text with special tokens added
        tokenized = self.tokenizer.tokenize(input_text, 1024)

        for _, token in enumerate(tokenized):
            self.assertIn(token, self.tokenizer.id_to_token)

        decoded_text = "".join([self.tokenizer.id_to_token[token] for token in tokenized])
        assert decoded_text.startswith(expected_text)
        assert decoded_text.endswith(self.tokenizer.padding_token)
  