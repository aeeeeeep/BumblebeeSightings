import random
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import ngrams_iterator
from torchtext.transforms import VocabTransform
from torchtext.vocab import vocab
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from utils import *
import warnings
warnings.filterwarnings("ignore")

class Args:
    def __init__(self) -> None:
        self.batch_size = 64
        self.lr = 0.1
        self.epochs = 12
        self.radio = 0.7
        self.num_workers = 12
        self.full_list = True

        self.embed_size = 100
        self.hidden_size = 16
        self.output_size = 2
        self.seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Img_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv.models.convnext_tiny(weights=tv.models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 2)
        self.out = nn.Softmax(dim=1)

    def forward(self, x): 
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

args = Args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class Text_Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(18440, args.embed_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(args.embed_size, args.output_size)

    def forward(self, text_token):
        embedded = self.embedding(text_token)
        pooled = nn.functional.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        out_put = self.linear(pooled)
        return out_put

    def get_embedding(self, token_list: list):
        return self.embedding(torch.Tensor(token_list).long())

class Text_Dataset(Dataset):
    def __init__(self, df, df_img, flag='train', max_length=20) -> None:
        df.set_index('GlobalID')
        df_img.set_index('GlobalID')
        df = pd.merge(df, df_img)
        df = df.dropna(subset=['FileName','Detection Date'])
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
        # data = df[['Notes', 'label']]
        self.text_list = df['Notes']
        self.flag = flag
        self.max_length = max_length
        assert self.flag in ['train', 'val'], 'not implement!'
        train_data, val_data = data_split(df, ratio=args.radio, shuffle=True)
        self.text_vocab, self.vocab_transform = self.reform_vocab(val_data['Notes'].to_list())
        self.text_label = val_data['label'].to_list()
        self.fast_data = self.generate_fast_text_data()
        self.val_data = val_data
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


transform_val = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([224, 224]),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

class Img_Dataset(Dataset):
    def __init__(self, path_list, labels):
        self.paths = path_list
        self.transform = transform_val
        self.labels = labels

    def __getitem__(self, index):
        path = self.paths[index]
        label = torch.zeros(2, dtype=torch.float32)
        label[self.labels[index]] = 1
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.paths)

def eval():
    df = pd.read_excel('./2021_MCM_Problem_C_Data/2021MCMProblemC_DataSet.xlsx')
    df_img = pd.read_excel('./2021_MCM_Problem_C_Data/2021MCM_ProblemC_Images_by_GlobalID.xlsx')
    train_dataset = Text_Dataset(df=df, df_img=df_img, flag='train')
    val_dataset = Text_Dataset(df=df, df_img=df_img, flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, drop_last=False)
    model = Text_Net(train_dataset.get_vocab_size())
    state_dict = torch.load('./checkpoint/fasttext_model_95.12_epoch_8.pth')
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    predict_list = []
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
            predict_list = predict_list + (list(np.array(pred.softmax(dim=1).cpu()[:,1],dtype=np.float32)))
            accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(),
                                        average='macro')
            recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(),
                                        F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
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
    print({
        "val_accuracy": avg_accuracy,
        "val_precision": avg_precision,
        "val_recall": avg_recall,
        "val_f1": avg_f1,
        "val_roc_auc": avg_roc_auc,
        "val_mse": avg_mse,
        "val_mae": avg_mae,
    })
    val_data = val_dataset.val_data.copy()
    val_data.insert(loc=1, column='text_label', value=predict_list)

    model = Img_Net()
    state_dict = torch.load('./checkpoint/model_epoch_10.pth')
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    path = './predict_img/'
    val_data = val_data.dropna(subset=['FileName','Detection Date'])
    val_data['file_exists'] = val_data['FileName'].apply(lambda x: os.path.exists(os.path.join(path, x)))
    val_data = val_data[val_data['file_exists']]
    val_data = val_data.drop('file_exists', axis=1)
    val_data['FileName'] = val_data['FileName'].apply(lambda x: path + x)
    img_list = val_data['FileName'].tolist()

    image_datasets = Img_Dataset(img_list, val_data['label'].tolist())
    image_loader = torch.utils.data.DataLoader(image_datasets, batch_size=16, num_workers=12)
    predict_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    mse_list = []
    mae_list = []
    with torch.no_grad():
        for idx, (img, labels) in enumerate(tqdm(image_loader)):
            pred = model(img.cuda())
            predict_list = predict_list + list(pred.cpu().numpy()[:,1])
            accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(),
                                        average='macro')
            recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(),
                                        F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
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
    print({
        "val_accuracy": avg_accuracy,
        "val_precision": avg_precision,
        "val_recall": avg_recall,
        "val_f1": avg_f1,
        "val_roc_auc": avg_roc_auc,
        "val_mse": avg_mse,
        "val_mae": avg_mae,
    })
    scaler = MinMaxScaler()
    val_data.insert(loc=1, column='img_label', value=predict_list)
    val_data['Detection Date'] = pd.to_datetime(val_data['Detection Date'])
    val_data['year'] = val_data['Detection Date'].dt.year.astype(float)
    val_data['month'] = val_data['Detection Date'].dt.month.astype(float)
    print("val_data:")
    print(val_data.head())
    val_data = val_data[['Latitude', 'Longitude', 'year', 'month', 'text_label', 'img_label', 'label']]
    data_norm = val_data.iloc[:,:-1].apply(lambda x: pd.Series(scaler.fit_transform(x.values.reshape(-1,1)).flatten(), index=x.index))
    print("data_norm:")
    print(data_norm.head())
    val_data = val_data[['Latitude', 'Longitude', 'year', 'month', 'text_label', 'img_label', 'label']]
    w = cal_weight(data_norm)
    w.columns = ['weight']
    weights = w.iloc[:,0].tolist()
    weighted_data = val_data.iloc[:, :6] * weights
    total_weight = weighted_data.sum(axis=1)
    preds = np.where(total_weight > 0.5, 1, 0)
    labels = val_data.iloc[:, 6]
    accuracy = (preds == labels).mean()
    print("accuracy: ", accuracy)


if __name__ == '__main__':
    eval()
