from transformers import BertTokenizer, BertForSequenceClassification

model_name = "bert-base-uncased"
save_dir = "./bert-base-uncased-local"

print("Downloading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

print("Saving to", save_dir)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("Done.")