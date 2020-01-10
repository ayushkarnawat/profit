# NOTE: Maybe we should have both a AA1/AA3 tokenizers (seperately)?
# Or rather, it should be smart enough to figure out if it is single or double.
# 
# TODO: Make special tokenizer for BERT sequences. Allows us to 
# represent different encoding/decodings for BERT-like tasks.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np

from profit.utils.data_utils.vocabs import IUPAC_AA1_VOCAB, IUPAC_AA3_VOCAB


class BaseTokenizer(ABC):
    """Base tokenizer."""

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError


class AminoAcidTokenizer(BaseTokenizer):
    """Tokenizer for amino acid sequences."""

    def __init__(self, vocab: str):
        if vocab == "iupac1":
            self.vocab: Dict[str, Any] = IUPAC_AA1_VOCAB
        elif vocab == "iupac3":
            self.vocab: Dict[str, Any] = IUPAC_AA3_VOCAB
        else:
            raise ValueError(f"{vocab} vocab type is unavailable.")
        self.flipped_vocab = {v:k for k,v in self.vocab.items()}
        self._vocab_type = vocab
        assert self.pad_token in self.vocab and self.unknown_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        if "<cls>" in self.vocab:
            return "<cls>"
        raise RuntimeError(f"{self._vocab_type} vocab does not support BERT " \
            "classification (start) token.")

    @property
    def stop_token(self) -> str:
        if "<sep>" in self.vocab:
            return "<sep>"
        raise RuntimeError(f"{self._vocab_type} vocab does not support seperation.")

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>" 
        raise RuntimeError(f"{self._vocab_type} vocab does not support masking.")

    @property
    def pad_token(self) -> str:
        return "<pad>"
    
    @property
    def unknown_token(self) -> str:
        return "<unk>"


    def tokenize(self, text: str) -> List[str]:
        """Tokenize (seperate) input."""
        return [x for x in text]


    def convert_token_to_id(self, token: str) -> int:
        """Converts a token (str/unicode) into an id using the vocab.
        
        NOTE: Tokens are case-sensitive, i.e. 'Ala' != 'ALA'.
        """
        try:
            return self.vocab[token]
        except KeyError:
            # We know that the unknown token has to exist, since we check for it
            # when the tokenizer is initialized.
            print(f"Unrecognized token `{token}`. Using {self.unknown_token} instead!")
            return self.vocab[self.unknown_token]


    def convert_id_to_token(self, id_: int) -> str:
        """Converts an id into its token representation using the vocab."""
        try:
            return self.flipped_vocab[id_]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{id_}'")


    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]


    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """Adds special tokens to the sequence. 
        
        Used for sequence classification tasks (specifically BERT-like 
        tasks). A BERT sequence has the following format: [CLS] X [SEP].
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token


    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text into ints (based off vocab)."""
        tokens = self.tokenize(text) if isinstance(text, str) else text
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, dtype=np.float)


    def decode(self, ids: List[int]) -> List[str]:
        """Decode ids back into tokens representation."""
        return [self.convert_id_to_token(id_) for id_ in ids]