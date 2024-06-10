from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return sentence_embedding

sentence = "My name is Salvatore, comme stai?"
embedding = get_sentence_embedding(sentence)