from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
import torch

# Load the fine-tuned model and tokenizer
model_name = "bigscience/mt0-small"
model_classifier_peft_name = "nelson2424/mt0-small-lora-finetune-grocery-action-classifier-v2"  # Replace with your fine-tuned model name or path
model_ner_peft_name = "nelson2424/mt0-small-lora-finetune-grocery-ner-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_classifier = AutoModelForSeq2SeqLM.from_pretrained(model_classifier_peft_name)
model_ner = AutoModelForSeq2SeqLM.from_pretrained(model_ner_peft_name)

input_text = """Identify the things that can be bought in a supermarket of the following text:
We need more coal"""
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
print(input_text)
# Perform inference
with torch.no_grad():
    outputs_class = model_classifier.generate(input_ids=input_ids, max_new_tokens=10)
    outputs_ner = model_ner.generate(input_ids=input_ids, max_new_tokens = 40)
    #print(outputs_class)
    print(tokenizer.batch_decode(outputs_class.detach().cpu().numpy(), skip_special_tokens=True))
    print(tokenizer.batch_decode(outputs_ner.detach().cpu().numpy(), skip_special_tokens=True))
