import torch.nn as nn
from transformers import BertModel
# from model.BANs.fc import FCNet
# from model.BANs.bilinearattention import BiAttention
from torchvision import models
import torch
from TorchCRF import CRF


class MnerModel(nn.Module):
    def __init__(self, config):
        super(MnerModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.visual_encoder = models.resnet50(pretrained=True)
        self.crf = CRF(len(self.config.tag2idx), batch_first=True)
        self.hidden2tag = nn.Linear(100, len(self.config.tag2idx))
        """BERT-BiLSTM-CNN"""
        self.lstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        # self.cnn = nn.Conv1d(in_channels=200, out_channels=100, kernel_size=1, stride=1)

    def forward(self, data, mode):

        x = data['x']
        crf_mask = x["attention_mask"]
        tags = data['label_ids']
        x = self.bert(**x)[0]

        if self.config.multi_modal:
            b_img = data['img']
            o = self.trans_img(b_img).view(-1, 2, 768)
            v_mask = torch.ones((self.config.batch_size, 2), requires_grad=False).bool().cuda()

        """BERT-BiLSTM-CNN"""
        x = self.lstm(x)[0]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        final_feature = x
        outputs = self.hidden2tag(final_feature)

        if mode == 'train':
            return -self.crf(outputs, tags, mask=crf_mask)
        elif mode == 'test':
            return self.crf.decode(outputs, mask=crf_mask)
        else:
            print('mode error! please choose train or test.')
