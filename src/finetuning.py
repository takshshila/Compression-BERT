from data_preprocessing import loadData
from transformers import BertForSequenceClassification, get_scheduler, AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from argparse import ArgumentParser
from torch import cuda

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
parser.add_argument('-e', '--epoch', nargs='+', default=3)
parser.add_argument('-lr', '--learning_rate', nargs='+', default=2e-5)
args = parser.parse_args()

model_ckpt = str(args.model_ckpt)
max_length = int(args.max_len)
batch_size = int(args.batch_size)
benchmark = args.data[0]
data = args.data[1]
epochs = int(args.epoch)
lr = int(args.learning_rate)

tokenizer, train_loader, eval_loader = loadData(model_ckpt, benchmark, data, batch_size)


def getModel(model_ckpt):
    model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)
    model.to(device)
    return model


def training():
    accelerator = Accelerator()
    model = getModel(model_ckpt)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, eval_loader, model, optimizer
    )

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
    torch.save(model, './test-mnli/model')
    torch.save(tokenizer, './test-mnli/tokenizer')


training()