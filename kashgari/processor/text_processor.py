# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: text_processor.py
# time: 12:27 下午

import operator
import collections
from typing import Generator
from kashgari.processor.abs_processor import ABCProcessor


class TextProcessor(ABCProcessor):
    def __init__(self, **kwargs):
        super(ABCProcessor, self).__init__(**kwargs)
        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')

        self.vocab2idx = {}
        self.idx2vocab = {}

    def build_vocab_dict(self, generator: Generator, min_count: int=3):
        generator.reset()
        if not self.vocab2idx:
            vocab2idx = {
                self.token_pad: 0,
                self.token_unk: 1,
                self.token_bos: 2,
                self.token_eos: 3
            }

            token2count = {}

            for x_set, _ in generator:
                for sentence in x_set:
                    for token in sentence:
                        count = token2count.get(token, 0)
                        token2count[token] = count + 1

            sorted_token2count = sorted(token2count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
            token2count = collections.OrderedDict(sorted_token2count)

            for token, token_count in token2count.items():
                if token not in vocab2idx and token_count >= min_count:
                    vocab2idx[token] = len(vocab2idx)
            self.vocab2idx = vocab2idx
            self.idx2vocab = dict([(v, k) for k, v in self.vocab2idx.items()])


if __name__ == "__main__":
    from kashgari.corpus import ChineseDailyNerCorpus
    from kashgari.utils import CorpusGenerator

    x, y = ChineseDailyNerCorpus.load_data()
    gen = CorpusGenerator(x, y)
    p = TextProcessor()
    p.build_vocab_dict(gen)
    print(p.vocab2idx)
