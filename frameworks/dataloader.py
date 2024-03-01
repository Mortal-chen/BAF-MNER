from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, config, text, img):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, do_lower_case=False)
        self.max_len = 512
        self.text_path = text
        self.img_path = img
        self.text = []
        self.label = []

        with open(self.text_path, 'r', encoding='utf-8') as f:
            line = f.read().split('\t')[1:]
            for idx, each in enumerate(line):
                len_id = len(str(idx))
                t = []
                l = []
                each = each.split('\n')[1:]
                for each_ in each:
                    each_ = each_.split(' ')

                    if len(each_) == 2:
                        t.append(each_[0])
                        l.append(each_[1])
                t = t[:-len_id]
                l = l[:-len_id]
                self.text.append(t)
                self.label.append(l)

        assert len(self.text) == len(self.label)
        if self.config.multi_modal == True:
            img_file = os.listdir(self.img_path)
            img_file.sort(key=lambda x: int(x[:-4]))
            self.pic_path = []
            for each_img in img_file:
                each_img_path = os.path.join(self.img_path, each_img)
                assert os.path.exists(each_img_path)
                self.pic_path.append(each_img_path)
        else:
            self.pic_path = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        s = self.text[idx]
        l = self.label[idx]
        tokens = ["[CLS]"]
        label_ids = [self.config.tag2idx["CLS"]]
        for word, label in zip(s, l):  # iterate every word
            token = self.tokenizer._tokenize(word)  # one word may be split into several tokens
            tokens.extend(token)
            for i, _ in enumerate(token):
                label_ids.append(self.config.tag2idx[label] if i == 0 else self.config.tag2idx["X"])
        tokens = tokens[:self.max_len - 1]
        tokens.append("[SEP]")
        label_ids = label_ids[:self.max_len - 1]
        label_ids.append(self.config.tag2idx["SEP"])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        mask = [1] * len(input_ids)
        segment_ids = [0] * self.max_len

        pad_len = self.max_len - len(input_ids)
        rest_pad = [0] * pad_len  # pad to max_len
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)
        tokens.extend(["pad"] * pad_len)

        # img process
        if self.pic_path is not None:
            img_path = self.pic_path[idx]
        else:
            img_path = None
        if len(tokens) != self.max_len:
            print('error')
        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids,
            "img_path": img_path,
        }


def collate_fn(batch):
    device = torch.device("cuda:0")
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []
    b_tokens = []
    pic_path = []
    img_list = []

    for idx, example in enumerate(batch):
        b_tokens.append(example["tokens"])
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
        pic_path.append(example['img_path'])

    # image process
    if pic_path[0] is not None:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        for img_path in pic_path:
            img = preprocess(Image.open(img_path).convert('RGB'))
            img_list.append(img)
        img_list_ = torch.stack(img_list, dim=0)
    else:
        img_list_ = []
    data = {
        "b_tokens": b_tokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device)
        },
        "label_ids": torch.tensor(label_ids).to(device),
        "img": torch.tensor(img_list_).to(device)
    }
    return data
