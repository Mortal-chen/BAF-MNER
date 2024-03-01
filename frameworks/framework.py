import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from frameworks.dataloader import MyDataset, collate_fn
from tqdm import tqdm
from model.model import MnerModel


def count_params(model):
    """计算模型参数"""
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)


class Framework(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        train_dataset = MyDataset(self.config, self.config.train_fn, self.config.img_path)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config.batch_size,
                                      collate_fn=collate_fn)
        dev_dataset = MyDataset(self.config, self.config.dev_fn, self.config.img_path)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=self.config.batch_size,
                                    collate_fn=collate_fn)
        test_dataset = MyDataset(self.config, self.config.test_fn, self.config.img_path)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.config.batch_size,
                                     collate_fn=collate_fn)
        print('===== ...data load done... =====')
        model = MnerModel(self.config).to(self.device)
        count_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

        res = {"best_f1": 0.0, "epoch": -1}
        start = time.time()
        for epoch in range(self.config.epoch):
            model.train()
            print("===== ...Epoch:{} Start training... =====".format(epoch))
            for i, data in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                loss = model(data, mode='train')
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                if i % self.config.log_fre == 0:
                    print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))
            scheduler.step()
            to_save = self.predict(epoch, model, dev_dataloader, mode="val", res=res)
            self.predict(epoch, model, test_dataloader, mode="test", res=res)
            if to_save:
                self.save_model(model, self.config.ckpt_path)

        print("================== train done! ================")
        end = time.time()
        hour = int((end - start) // 3600)
        minute = int((end - start) % 3600 // 60)
        print("total time: {} hour - {} minute".format(hour, minute))

    def predict(self, epoch, model, dataloader, mode="val", res=None):
        model.eval()
        with torch.no_grad():
            filepath = self.config.predict_file.format(mode, epoch)
            with open(filepath, "w", encoding="utf8") as fw:
                for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
                    y = data['label_ids']
                    b_tokens = data['b_tokens']
                    output = model(data, mode='test')

                    # write into file
                    for idx, pre_seq in enumerate(output):
                        ground_seq = y[idx]
                        for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                            if ground_idx == self.config.tag2idx["PAD"] or ground_idx == self.config.tag2idx["X"] \
                                    or ground_idx == self.config.tag2idx["CLS"] \
                                    or ground_idx == self.config.tag2idx["SEP"]:
                                continue
                            else:
                                predict_tag = self.config.idx2tag[pre_idx] if self.config.idx2tag[pre_idx] not in [
                                    "PAD", "X", "CLS", "SEP"] else "O"
                                true_tag = self.config.idx2tag[ground_idx.data.item()]
                                line = "{}\t{}\t{}\n".format(b_tokens[idx][pos], predict_tag, true_tag)
                                fw.write(line)
            print("=============={} -> {} epoch eval done=================".format(mode, epoch))
            cur_f1 = self.evaluate_pred_file(filepath)
            to_save = False
            if mode == "val":
                if res["best_f1"] < cur_f1:
                    res["best_f1"] = cur_f1
                    res["epoch"] = epoch
                    to_save = True
                print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))
            return to_save

    def evaluate_pred_file(self, filepath):
        labels_pred, labels = self.get_seq(filepath, self.config.tag2idx)
        labels_pred = [labels_pred]
        labels = [labels]
        acc, f1, p, r = self.evaluate(labels_pred, labels, self.config.tag2idx)
        print("overall: ", acc, p, r, f1)
        arr = ['disease', 'symptom', 'organ', 'attribute', 'treatment']
        for class_type in arr:
            class_f1, class_p, class_r = self.evaluate_each_class(labels_pred, labels, self.config.tag2idx, class_type)
            print(class_type, class_p, class_r, class_f1)
        return f1

    def get_seq(self, filepath, tag2idx):  # 获取预测文件的label，并返回成数字形式
        pred = []
        ground = []
        with open(filepath, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                word, pl, gl = line.split("\t")
                pred.append(tag2idx[pl])
                ground.append(tag2idx[gl])
        return pred, ground

    def evaluate(self, labels_pred, labels, tags):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for lab, lab_pred in zip(labels, labels_pred):
            lab = lab
            lab_pred = lab_pred
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            lab_chunks = set(self.get_chunks(lab, tags))
            lab_pred_chunks = set(self.get_chunks(lab_pred, tags))
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return acc, f1, p, r

    def get_chunks(self, seq, tags):
        default = tags['O']
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def get_chunk_type(self, tok, idx_to_tag):
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def evaluate_each_class(self, labels_pred, labels, tags, class_type):
        correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

        for lab, lab_pred in zip(labels, labels_pred):
            lab_pre_class_type = []
            lab_class_type = []

            lab = lab
            lab_pred = lab_pred
            lab_chunks = self.get_chunks(lab, tags)
            lab_pred_chunks = self.get_chunks(lab_pred, tags)
            for i in range(len(lab_pred_chunks)):
                if lab_pred_chunks[i][0] == class_type:
                    lab_pre_class_type.append(lab_pred_chunks[i])
            lab_pre_class_type_c = set(lab_pre_class_type)

            for i in range(len(lab_chunks)):
                if lab_chunks[i][0] == class_type:
                    lab_class_type.append(lab_chunks[i])
            lab_class_type_c = set(lab_class_type)

            lab_chunksss = set(lab_chunks)
            correct_preds_cla_type += len(lab_pre_class_type_c & lab_chunksss)
            total_preds_cla_type += len(lab_pre_class_type_c)
            total_correct_cla_type += len(lab_class_type_c)

        p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
        r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0

        return f1, p, r

    def save_model(self, model, model_path=None):
        torch.save(model.state_dict(), model_path)
        print("Current Best MNER model has beed saved!")
