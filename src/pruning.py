import torch
import torch.nn.utils.prune as prune
from torch import cuda
import os

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Device: {device}")

def pruning(bert_model, pamount):
    #bert_model = torch.load(modelPath, map_location=device)

    parameters_to_prune = ()
    layers = len(bert_model.bert.encoder.layer)
    for i in range(layers):
        parameters_to_prune += (
            (bert_model.bert.encoder.layer[i].attention.self.key, 'weight'),
            (bert_model.bert.encoder.layer[i].attention.self.query, 'weight'),
            (bert_model.bert.encoder.layer[i].attention.self.value, 'weight'),
        )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pamount,
    )
    
    check_model_sparcity(bert_model)
    #torch.save(model, '/home/trawat2/LearningProject/SavedTraining/MNLI/PruneBERT/model')
    '''
    out_path='/home/trawat2/LearningProject/SavedTraining/MNLI/PruneBERT/'+ str(pamount)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    #torch.save(bert_model, out_path)
    torch.save(bert_model, out_path+'/model')
    '''
    
    return bert_model

def check_model_sparcity(bert_model):
    layers = len(bert_model.bert.encoder.layer)
    for i in range(layers):
        print(
            "Sparsity in Layer {}-th key weight: {:.2f}%".format(
                i + 1,
                100. * float(torch.sum(bert_model.bert.encoder.layer[i].attention.self.key.weight == 0))
                / float(bert_model.bert.encoder.layer[i].attention.self.key.weight.nelement())
            )
        )
        print(
            "Sparsity in Layer {}-th query weightt: {:.2f}%".format(
                i + 1,
                100. * float(torch.sum(bert_model.bert.encoder.layer[i].attention.self.query.weight == 0))
                / float(bert_model.bert.encoder.layer[i].attention.self.query.weight.nelement())
            )
        )
        print(
            "Sparsity in Layer {}-th value weight: {:.2f}%".format(
                i + 1,
                100. * float(torch.sum(bert_model.bert.encoder.layer[i].attention.self.value.weight == 0))
                / float(bert_model.bert.encoder.layer[i].attention.self.value.weight.nelement())
            )
        )
        print()

    numerator, denominator = 0, 0
    for i in range(layers):
        numerator += torch.sum(bert_model.bert.encoder.layer[i].attention.self.key.weight == 0)
        numerator += torch.sum(bert_model.bert.encoder.layer[i].attention.self.query.weight == 0)
        numerator += torch.sum(bert_model.bert.encoder.layer[i].attention.self.value.weight == 0)

        denominator += bert_model.bert.encoder.layer[i].attention.self.key.weight.nelement()
        denominator += bert_model.bert.encoder.layer[i].attention.self.query.weight.nelement()
        denominator += bert_model.bert.encoder.layer[i].attention.self.value.weight.nelement()

    print("Global sparsity: {:.2f}%".format(100. * float(numerator) / float(denominator)))
    

#pruning('/home/trawat2/LearningProject/SavedTraining/MNLI/model')
