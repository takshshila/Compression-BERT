from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
#tokenizer=None

def loadData(tokenizer, model_ckpt, benchmark, data, batch_size):
    #tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["hypothesis"],
            examples["premise"],
            # max_length=max_length,
            truncation=True
        )
        return tokenized_examples
    dataset = load_dataset(benchmark, data)

    dataset = dataset.map(prepare_train_features, batched=True)
    dataset = dataset.remove_columns(['premise', 'hypothesis', 'idx'])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    #print(dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(dataset['validation_matched'], batch_size=batch_size, collate_fn=data_collator)
    return  train_loader, eval_loader


#loadData()



