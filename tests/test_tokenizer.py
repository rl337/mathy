# tests for tokenizer
import unittest
from mathy.tokenizer import CharacterTokenizer

class TokenizerTest(unittest.TestCase):

    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_character_tokenizer(self):
        corpus = ["This is an example sentence", "Byte Pair Encoding is interesting", "Let's tokenize text"]
        self.tokenizer.train(corpus)

        tokenized = self.tokenizer.tokenize("This is an example sentence")
        self.assertEqual(tokenized, [1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8,6, 9, 10, 7, 11, 12, 13, 9, 6, 5, 9, 8, 14, 9, 8, 15, 9, 6, 16])
  