from data_preprocessing import loadData, count_params
from pruning import pruning, check_model_sparcity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, \
    get_scheduler, AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from argparse import ArgumentParser
from torch import cuda
import os

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
parser.add_argument('-b', '--batch_size', nargs='+', default=8)
parser.add_argument('-e', '--epoch', nargs='+', default=10)
parser.add_argument('-lr', '--learning_rate', nargs='+', default=2e-5)
args = parser.parse_args()

model_ckpt = "gchhablani/bert-base-cased-finetuned-mnli"  # M-FAC/bert-mini-finetuned-mnli" #prajjwal1/bert-mini" #str(args.model_ckpt)
max_length = int(args.max_len)
batch_size = int(args.batch_size)
benchmark = args.data[0]
data = args.data[1]
epochs = 3  # int(args.epoch)
lr = int(args.learning_rate)

# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# tokenizer = AutoTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")
# out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/' +model_ckpt+ str(epochs)

origin_path = '/home/trawat2/LearningProject/SavedTraining/MNLI/gchhablani/bert-base-cased-finetuned-mnli10'
# origin_path='/home/trawat2/LearningProject/SavedTraining/MNLI/' +model_ckpt+ str(epochs)
tokenizer = torch.load(origin_path + '/tokenizer', map_location=device)

# tokenizer, train_loader, eval_loader = loadData(model_ckpt, benchmark, data, batch_size)
train_loader, eval_loader = loadData(tokenizer, model_ckpt, benchmark, data, batch_size)

accelerator = Accelerator()


def getModel(model_ckpt):
    # model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)

    mpath = origin_path + "/PruneBERT/0.3"
    # print("Pruned model form :: "+mpath)
    model = torch.load(mpath + '/model', map_location=device)
    model.to(device)

    return model


def getTokenizer(model_ckpt):
    # model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)
    print("Tokenizer form :: " + origin_path)
    tokenizer = torch.load(origin_path + '/tokenizer', map_location=device)

    return tokenizer


def training():
    print(" Training: " + model_ckpt)
    accelerator = Accelerator()
    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # train_loader, eval_loader = loadData(tokenizer, model_ckpt, benchmark, data, batch_size)
    model = getModel(model_ckpt)
    tokenizer = getTokenizer(model_ckpt)
    # model = torch.load('/home/trawat2/LearningProject/SavedTraining/MNLI/model', map_location=device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, eval_loader, model, optimizer
    )
    # Clearing cache
    # gc.collect()
    # torch.cuda.empty_cache()

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
    check_model_sparcity(model)

    evaluation(model, tokenizer)

    # out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/' +model_ckpt+ str(epochs)
    out_path = origin_path + '/' + model_ckpt + str(epochs)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    torch.save(model, out_path + '/model')
    torch.save(tokenizer, out_path + '/tokenizer')

    return model, tokenizer


def evaluation(model, tokenizer):
    # print("Evaluations::::"+origin_path)
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()

    '''
    if model == None:
        mpath = '/home/trawat2/LearningProject/SavedTraining/MNLI/model'
        model = torch.load(mpath, map_location=device)

    if tokenizer==None:
        tpath = '/home/trawat2/LearningProject/SavedTraining/MNLI/tokenizer'
        tokenizer = torch.load(tpath, map_location=device)
    '''

    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # train_loader, eval_loader = loadData(tokenizer, model_ckpt, benchmark, data, batch_size)

    # print("model loaded successfully....")
    # print(model)

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
    print(f"RAM used before eval: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

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


# training()
# evaluation()

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


if __name__ == '__main__':
    # origin_path = '/home/trawat2/LearningProject/SavedTraining/MNLI'
    # Clearing cache
    model, tokenizer = training()

    '''

    #model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
    #evaluation(model,tokenizer)
    model = torch.load(origin_path+'/model', map_location=device)
    model.to(device)

    gc.collect()
    torch.cuda.empty_cache()


    pamount=0.3
    prunedModel = pruning(model, pamount)
    print("Pruned model form: "+str(pamount))
    evaluation(prunedModel,tokenizer)
    #model, tokenizer = training()
    print("Pruned Model size")
    print_size_of_model(prunedModel)




    print(origin_path)
    model = torch.load(origin_path+'/model', map_location=device)
    model.to(device)


    #evaluation(model,tokenizer)



    mpath = origin_path+"/PruneBERT/0.3"
    print("Pruned model form :: "+mpath)

    prunedModel = torch.load(mpath+'/model', map_location=device)
    prunedModel.to(device)

    check_model_sparcity(prunedModel)
    evaluation(prunedModel,tokenizer)
     '''

    # Training pruned model

    '''

    #model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
    out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/' +model_ckpt+ str(epochs)
    model = torch.load(out_path+'/model', map_location=device)

    #pamount=0.3
    #prunePath= out_path+'/PruneBERT/'+ str(pamount)

    #model = torch.load(prunePath+'/model', map_location=device)
    #check_model_sparcity(model)

    evaluation(model,tokenizer)

    #model = AutoModelForSequenceClassification.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")
    #model.to(device)
    #evaluation(model,tokenizer)


    #torch.save(model, out_path+'/model')
    #torch.save(tokenizer, out_path+'/tokenizer')


    # Clearing cache
    #gc.collect()
    #torch.cuda.empty_cache()



    #mpath = '/home/trawat2/LearningProject/SavedTraining/MNLI/pruneBert/pruneBERT'
    #print("Evaluating Pruned Model")
    #mpath = '/home/trawat2/LearningProject/SavedTraining/MNLI/model'
    #model = torch.load(mpath, map_location=device)

    #prunedModel = pruning(model)
    #evaluation(mpath,tpath)
    #evaluation(model)
    '''

    '''
    out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/' +model_ckpt+ str(epochs)
    #model = torch.load(out_path+'/model', map_location=device)


    pamount=0.3
    #prunedModel = pruning(model, pamount)

    prunePath= out_path+'/PruneBERT/'+ str(pamount)


    if not os.path.exists(prunePath):
        os.makedirs(prunePath)
    torch.save(prunedModel, prunePath+'/model')




    path = "/home/trawat2/LearningProject/SavedTraining/MNLI/M-FAC/bert-mini-finetuned-mnli10"
    pmpath =path+"/PruneBERT/0.3/model"
    if os.path.exists(pmpath):
        print("dir Exists")


    p_model = torch.load(pmpath, map_location=device)
    #model = torch.load(mpath, map_location=device)
    print("Pruned Moded....")
    print(p_model)


    print(" Evaluation Pruned model: " +model_ckpt+ str(epochs))

    pmpath =out_path+"/PruneBERT/0.3/model"
    p_model = torch.load(pmpath, map_location=device)

    tpath=out_path+'/tokenizer'

    tokenizer = torch.load(tpath, map_location=device)

    evaluation(p_model,tokenizer)
    #evaluation(model,tokenizer)

    '''

    '''

    tokenizer = AutoTokenizer.from_pretrained("M-FAC/bert-mini-finetuned-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("M-FAC/bert-mini-finetuned-mnli")

    #out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/PruneBERT/model'
    #prunedModel = pruning(model, 0.2)
    pamount = 0.2
    out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/PruneBERT/'+ str(pamount)+'/model'
    prunedModel = torch.load(out_path, map_location=device)
    evaluation(prunedModel,tokenizer)
    '''

    '''
    prunedModel = pruning(model, 0.3)
    evaluation(prunedModel,tokenizer)

    prunedModel = pruning(model, 0.4)
    evaluation(prunedModel,tokenizer)

    prunedModel = pruning(model, 0.4)
    evaluation(prunedModel,tokenizer)
    '''