WANDB = False
import random
from collections import Counter, OrderedDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from torchtext.data.utils import ngrams_iterator
from torchtext.transforms import VocabTransform
from torchtext.vocab import vocab
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score

from utils import *
import wandb
import warnings
warnings.filterwarnings("ignore")

if WANDB:
    wandb.init(
            project="MultimodalCommentAnalysis",
            name="north-fasttext",
            )   

class Args:
    def __init__(self) -> None:
        self.batch_size = 64
        self.lr = 0.1
        self.epochs = 12
        self.radio = 0.7
        self.num_workers = 12
        self.full_list = False

        self.embed_size = 100
        self.hidden_size = 16
        self.output_size = 2
        self.seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = Args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, args.embed_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(args.embed_size, args.output_size)

    def forward(self, text_token):
        embedded = self.embedding(text_token)
        # embedded = embedded + torch.randn(embedded.shape).cuda() * 0.1
        # embedded = self.dropout(embedded)
        pooled = nn.functional.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        out_put = self.linear(pooled)
        return out_put

    def get_embedding(self, token_list: list):
        return self.embedding(torch.Tensor(token_list).long())

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Dataset(Dataset):
    def __init__(self, df, df_img, flag='train', max_length=20) -> None:
        df.set_index('GlobalID')
        df_img.set_index('GlobalID')
        df.drop(df[df['Notes'] == ' '].index, inplace=True)
        df = pd.concat([df[df['Lab Status'] == 'Negative ID'], df[df['Lab Status'] == 'Positive ID']])
        df['label'] = df['Lab Status'].apply(lambda i: 0 if i == 'Negative ID' else 1)
        df['Notes'] = df['Notes'].apply(str)
        df.drop_duplicates(subset=['Notes'], inplace=True)
        with open("./2021_MCM_Problem_C_Data/aug.txt") as f:
            for i in f.readlines():
                df = df.append({'Notes':i, 'label':int(1)}, ignore_index=True)
        df['Notes_token'] = process_notes(df['Notes'])
        df.drop(df[df['Notes_token'].apply(lambda x: len(x) < 3)].index, inplace=True)
        data = df[['Notes', 'label']]
        self.text_list = data['Notes']
        self.flag = flag
        self.max_length = max_length
        assert self.flag in ['train', 'val'], 'not implement!'
        train_data, val_data = data_split(data, ratio=args.radio, shuffle=True)
        if self.flag == 'train':
            self.text_vocab, self.vocab_transform = self.reform_vocab(train_data['Notes'].to_list())
            self.text_label = train_data['label'].to_list()
            self.fast_data = self.generate_fast_text_data()
            self.len = len(train_data)

        else:
            self.text_vocab, self.vocab_transform = self.reform_vocab(val_data['Notes'].to_list())
            self.text_label = val_data['label'].to_list()
            self.fast_data = self.generate_fast_text_data()
            self.len = len(val_data)

    def __getitem__(self, index):
        data_row = self.fast_data[index]
        data_row = pad_or_cut(data_row, self.max_length)
        data_label = torch.zeros(2, dtype=torch.float32)
        data_label[self.text_label[index]] = 1
        return data_row, data_label

    def __len__(self) -> int:
        return self.len

    def get_labels(self):
        return self.text_label

    def reform_vocab(self, text_list):
        total_word_list = []
        for _ in text_list:
            total_word_list += list(ngrams_iterator(_.split(" "), 2))
        counter = Counter(total_word_list)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        special_token = ["<UNK>", "<SEP>"]
        text_vocab = vocab(ordered_dict, specials=special_token)
        text_vocab.set_default_index(0)
        vocab_transform = VocabTransform(text_vocab)
        return text_vocab, vocab_transform

    def generate_fast_text_data(self):
        fast_data = []
        for sentence in self.text_list:
            all_sentence_words = list(ngrams_iterator(sentence.split(' '), 2))
            sentence_id_list = np.array(self.vocab_transform(all_sentence_words))
            fast_data.append(sentence_id_list)
        return fast_data

    def get_vocab_transform(self):
        return self.vocab_transform

    def get_vocab_size(self):
        return len(self.text_vocab)


def train():
    df = pd.read_excel('./2021_MCM_Problem_C_Data/2021MCMProblemC_DataSet.xlsx')
    df_img = pd.read_excel('./2021_MCM_Problem_C_Data/2021MCM_ProblemC_Images_by_GlobalID.xlsx')
    train_dataset = Dataset(df=df, df_img=df_img, flag='train')
    train_dataloader = DataLoaderX(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))
    val_dataset = Dataset(df=df, df_img=df_img, flag='val')
    val_dataloader = DataLoaderX(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, drop_last=True)

    model = Net(train_dataset.get_vocab_size()).to(args.device)
    corss_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    best_acc = 0.

    for epoch in range(args.epochs):
        print('Epoch: ', epoch)
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        mse_list = []
        mae_list = []

        model.train()
        for idx, (data, target) in enumerate(tqdm(train_dataloader)):
            data, labels = data.to(args.device), target.to(args.device)
            pred = model(data)
            loss = corss_loss(pred, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if WANDB:
                wandb.log({"train_loss": loss.item(),})
            accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1]) 
            except ValueError:
                pass
            mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            mse_list.append(mse)
            mae_list.append(mae)

        avg_accuracy = np.mean(accuracy_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_roc_auc = np.mean(roc_auc_list)
        avg_mse = np.mean(mse_list)
        avg_mae = np.mean(mae_list)
        if WANDB:
            wandb.log({
                "train_accuracy":avg_accuracy,
                "train_precision":avg_precision,
                "train_recall":avg_recall,
                "train_f1":avg_f1,
                "train_roc_auc":avg_roc_auc,
                "train_mse":avg_mse,
                "train_mae":avg_mae,
                })
        print({
            "train_accuracy":avg_accuracy,
            "train_precision":avg_precision,
            "train_recall":avg_recall,
            "train_f1":avg_f1,
            "train_roc_auc":avg_roc_auc,
            "train_mse":avg_mse,
            "train_mae":avg_mae,
            })

        model.eval()
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        mse_list = []
        mae_list = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(val_dataloader)):
                data, labels = data.to(args.device), target.to(args.device)
                pred = model(data)
                loss = corss_loss(pred, labels)
                accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
                precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                try:
                    roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
                except ValueError:
                    pass
                mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
                mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())

                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                roc_auc_list.append(roc_auc)
                mse_list.append(mse)
                mae_list.append(mae)
        avg_accuracy = np.mean(accuracy_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_roc_auc = np.mean(roc_auc_list)
        avg_mse = np.mean(mse_list)
        avg_mae = np.mean(mae_list)
        if WANDB:
            wandb.log({
                "val_accuracy":avg_accuracy,
                "val_precision":avg_precision,
                "val_recall":avg_recall,
                "val_f1":avg_f1,
                "val_roc_auc":avg_roc_auc,
                "val_mse":avg_mse,
                "val_mae":avg_mae,
                })
        print({
            "val_accuracy":avg_accuracy,
            "val_precision":avg_precision,
            "val_recall":avg_recall,
            "val_f1":avg_f1,
            "val_roc_auc":avg_roc_auc,
            "val_mse":avg_mse,
            "val_mae":avg_mae,
            })


        if avg_accuracy > best_acc:
            print('Save ...')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(),
                    './checkpoint/fasttext_model_{:.2f}_epoch_{}.pth'.format(100 * avg_accuracy, epoch))
            best_acc = avg_accuracy

if __name__ == '__main__':
    train()
