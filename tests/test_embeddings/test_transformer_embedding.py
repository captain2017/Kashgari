# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_transformer_embedding.py
# time: 2:47 下午

import unittest
from tensorflow.keras.utils import get_file
from kashgari.macros import DATA_PATH
from kashgari.embeddings import BertEmbedding, TransformerEmbedding
from kashgari.tasks.labeling import BiLSTM_Model
from kashgari.tasks.classification import BiLSTM_Model as Classification_BiLSTM_Model

from tests.test_macros import TestMacros


class TestTransferEmbedding(unittest.TestCase):

    def test_bert_embedding(self):
        bert_path = get_file('bert_sample_model',
                             "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                             cache_dir=DATA_PATH,
                             untar=True)
        embedding = BertEmbedding(model_folder=bert_path, sequence_length=12)

        # --- classification ----
        x, y = TestMacros.load_classification_corpus()
        embedding.embed(x)

        model = Classification_BiLSTM_Model(embedding=embedding)
        model.fit(x, y, epochs=1)

        # ------ labeling -------
        x, y = TestMacros.load_labeling_corpus()

        model = BiLSTM_Model(embedding=embedding)
        model.fit(x, y, epochs=1)


if __name__ == "__main__":
    pass
