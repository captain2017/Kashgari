# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: generator.py
# time: 4:53 下午

import random
from types import TracebackType
from typing import List, Generator, Type, Optional
from kashgari.utils import get_list_subset


class CorpusGenerator:

    def __init__(self, x: List, y: List):
        self.x = x
        self.y = y

        self._index_list = list(range(len(self.x)))
        self._current_index = 0

        random.shuffle(self._index_list)

    def reset(self):
        self._current_index = 0

    @property
    def steps(self) -> int:
        return len(self.x)

    def __iter__(self):
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index >= len(self.x) - 1:
            raise StopIteration()

        sample_index = self._index_list[self._current_index]
        return self.x[sample_index], self.y[sample_index]

    def __call__(self, *args, **kwargs):
        return self


class BatchDataGenerator:
    def __init__(self, corpus, text_processor, label_processor, batch_size=64):
        self.corpus = corpus
        self.text_processor = text_processor
        self.label_processor = label_processor

        self.batch_size = batch_size

    @property
    def steps(self) -> int:
        return self.corpus.steps // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        x_set = []
        y_set = []
        for i in range(self.batch_size):
            try:
                x, y = next(self.corpus)
            except StopIteration:
                self.corpus.reset()
                x, y = next(self.corpus)
            x_set.append(x)
            y_set.append(y)

        return self.text_processor.numerize_samples(x_set), \
               self.label_processor.numerize_samples(y_set)

    def __call__(self, *args, **kwargs):
        return self


if __name__ == "__main__":
    pass
