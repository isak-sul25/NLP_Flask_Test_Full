import pandas as pd
import numpy as np
import random
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, TrainingArguments, Trainer, EvalPrediction, AutoModelForSequenceClassification, \
    pipeline


label_dic = {0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'curiosity',
             6: 'disappointment', 7: 'disapproval', 8: 'gratitude', 9: 'joy', 10: 'love', 11: 'optimism',
             12: 'realization', 13: 'neutral'}

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 2023
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv("data.csv")

    # Distribution of a df
    def distribution(datafr):
        dist = {}
        lngth = len(datafr.index)

        for column in list(label_dic.values()):
            dist[column] = datafr[column].sum() / lngth

        return dist

    dist = distribution(df)
    names = list(dist.keys())
    values = list(dist.values())

    dist_df = pd.DataFrame({"Names": names,
                            "Dist": values})

    dist_df = dist_df.sort_values('Dist', ascending=False)

    # Split dataset into train, validation and test (80:10:10)
    hf_ds = Dataset.from_pandas(df)
    ds_train_test_valid = hf_ds.train_test_split(test_size=0.2, shuffle=True)
    ds_test_valid = ds_train_test_valid['test'].train_test_split(test_size=0.5, shuffle=True)

    ds_dict = DatasetDict({
        'train': ds_train_test_valid['train'],
        'validation': ds_test_valid['train'],
        'test': ds_test_valid['test']
    })

    labels = [label for label in ds_dict['train'].features.keys() if label not in ['id', 'text']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Converts HuggingFace predictions to an appropriate format
    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def to_int_pred(predictions, labels, threshold):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))

        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)

        # When not greater than 0, pick only the 1 (the highest) label; (multi-class)
        if threshold > 0:
            y_pred[np.where(probs >= threshold)] = 1
        else:
            y_pred[np.arange(len(probs)), probs.argmax(1)] = 1

        y_true = labels

        return y_true, y_pred

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        y_true, y_pred = to_int_pred(predictions, labels, threshold)

        # compute metrics
        precision_micro_score = precision_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        recall_micro_score = recall_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        # return as dictionary
        metrics = {'accuracy': accuracy,
                   'precision (micro)': precision_micro_score,
                   'recall (micro)': recall_micro_score,
                   'f1 (micro)': f1_micro_average,
                   'f1 (macro)': f1_macro_average,
                   }
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    # Score Function for the HuggingFace models
    def score(y_true, y_pred, index):
        precision_micro_score = precision_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        recall_micro_score = recall_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {'accuracy': accuracy,
                   'precision (micro)': precision_micro_score,
                   'recall (micro)': recall_micro_score,
                   'f1 (micro)': f1_micro_average,
                   'f1 (macro)': f1_macro_average,
                   }
        return pd.DataFrame(metrics, index=[index])

    tr = ds_dict["train"].to_pandas()
    va = ds_dict["validation"].to_pandas()
    te = ds_dict["test"].to_pandas()

    X_train = tr["text"]
    y_train = tr[labels].to_numpy()

    X_val = va["text"]
    y_val = va[labels].to_numpy()

    X_test = te["text"]
    y_test = te[labels].to_numpy()

    # Model
    distilbert_opt2 = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                         problem_type="multi_label_classification",
                                                                         num_labels=len(labels),
                                                                         id2label=id2label,
                                                                         label2id=label2id)

    # Tokenizer
    distilbert_tokenizer2 = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    ## Encodes the dataset to be used for the model
    def preprocess_data(examples):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = distilbert_tokenizer2(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    # Remove text and labels columns and keep 'input_ids', 'attention_mask', 'labels'
    encoded_dataset = ds_dict.map(preprocess_data, batched=True, remove_columns=ds_dict['train'].column_names)
    encoded_dataset.set_format("torch")

    args = TrainingArguments(
        output_dir=f"distilbert-base-uncased_opt2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        distilbert_opt2,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=distilbert_tokenizer2,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model = distilbert_opt2.to('cpu')
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=distilbert_tokenizer2)

    return classifier
