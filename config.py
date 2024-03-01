
class Config(object):
    def __init__(self):

        self.batch_size = 8
        self.epoch = 200
        self.lr = 0.005
        self.log_fre = 100

        self.train_fn = 'data/text_6995/train_6995.txt'
        self.dev_fn = 'data/text_6995/dev_6995.txt'
        self.test_fn = 'data/text_6995/test_6995.txt'
        self.bert_path = './pretrained/bert-base-chinese'
        self.img_path = 'data/img'
        self.predict_file = './output/med/{}/epoch_{}.txt'
        self.ckpt_path = './ckpt/best_model.pt'
        self.multi_modal = True   # 是否为多模态模型

        self.tag2idx = {
            "PAD": 0,
            "B-disease": 1,
            "I-disease": 2,
            "B-symptom": 3,
            "I-symptom": 4,
            "B-organ": 5,
            "I-organ": 6,
            "B-attribute": 7,
            "I-attribute": 8,
            "B-treatment": 9,
            "I-treatment": 10,
            "O": 11,
            "X": 12,
            "CLS": 13,
            "SEP": 14
        }

        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
