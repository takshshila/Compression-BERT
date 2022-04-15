from data_preprocessing import loadData
from transformers import AutoTokenizer, BertForSequenceClassification, get_scheduler, AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from argparse import ArgumentParser
from torch import cuda

from datasets import load_metric
import gc
import psutil
import time

device = 'cuda' if cuda.is_available() else 'cpu'

print(f"Device: {device}")

parser = ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', default=['glue', 'mnli'])
parser.add_argument('-m', '--model_ckpt', nargs='+', default='bert-base-cased')
parser.add_argument('-l', '--max_len', nargs='+', default=100)
parser.add_argument('-b', '--batch_size', nargs='+', default=16)
parser.add_argument('-e', '--epoch', nargs='+', default=8)
parser.add_argument('-lr', '--learning_rate', nargs='+', default=2e-5)
args = parser.parse_args()

model_ckpt = str(args.model_ckpt)
max_length = int(args.max_len)
batch_size = int(args.batch_size)
benchmark = args.data[0]
data = args.data[1]
epochs = int(args.epoch)
lr = int(args.learning_rate)

# tokenizer, train_loader, eval_loader = loadData(model_ckpt, benchmark, data, batch_size)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
train_loader, eval_loader = loadData(tokenizer, model_ckpt, benchmark, data, batch_size)

accelerator = Accelerator()


def getModel(model_ckpt):
    model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)
    model.to(device)
    return model


def training():
    accelerator = Accelerator()
    # model = getModel(model_ckpt)
    model = torch.load('/home/trawat2/LearningProject/SavedTraining/MNLI/model', map_location=device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, eval_loader, model, optimizer
    )
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()

    num_epochs = epochs
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progressBar = tqdm(range(num_training_steps))

    model.train()

    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()

    # start time
    startTime = time.time()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progressBar.update(1)

    # print(num_training_steps)
    print("--- %s seconds ---" % (time.time() - startTime))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    torch.save(model, '/home/trawat2/LearningProject/SavedTraining/MNLI/' + str(epochs) + '/model')
    torch.save(tokenizer, '/home/trawat2/LearningProject/SavedTraining/MNLI/' + str(epochs) + '/tokenizer')


def evaluation():
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    mpath = '/home/trawat2/LearningProject/SavedTraining/MNLI/' + str(epochs) + '/model'
    model = torch.load(mpath, map_location=device)
    print("model loaded successfully....")
    print(model)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, eval_loader, model, optimizer)

    # metric = load_metric("glue", "mnli")

    metric_acc = load_metric("glue", "mnli")
    metric_f1 = load_metric("f1")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")

    model.eval()

    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()

    # start time
    startTime = time.time()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # metric.add_batch(predictions=predictions, references=batch["labels"])
        # metric.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))
        metric_acc.add_batch(predictions=accelerator.gather(predictions),
                             references=accelerator.gather(batch["labels"]))
        metric_f1.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))
        metric_precision.add_batch(predictions=accelerator.gather(predictions),
                                   references=accelerator.gather(batch["labels"]))
        metric_recall.add_batch(predictions=accelerator.gather(predictions),
                                references=accelerator.gather(batch["labels"]))

    acc_score = metric_acc.compute()
    f1_score = metric_f1.compute(average="weighted")
    precision_score = metric_precision.compute(average="weighted")
    metric_recall = metric_recall.compute(average="weighted")

    print('accuracy ::::: ', acc_score)
    print('f1_score ::::: ', f1_score)
    print('precision_score ::::: ', precision_score)
    print('metric_recall ::::: ', metric_recall)

    print("--- %s seconds ---" % (time.time() - startTime))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


training()
evaluation()

