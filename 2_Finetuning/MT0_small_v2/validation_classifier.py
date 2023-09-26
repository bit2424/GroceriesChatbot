from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
import torch

# Load the fine-tuned model and tokenizer
model_name = "bigscience/mt0-small"
model_classifier_peft_name = "nelson2424/mt0-small-lora-finetune-grocery-action-classifier"  # Replace with your fine-tuned model name or path
dataset_name = "nelson2424/Grocery_chatbot_text_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_classifier = AutoModelForSeq2SeqLM.from_pretrained(model_classifier_peft_name)

dataset_validation = load_dataset(dataset_name,split=['train[90%:100%]'])

eval_preds = []


input_ids = tokenizer(dataset_validation[0]["text"], return_tensors="pt", max_length=70 ,padding="max_length", truncation=True)["input_ids"]

# Perform inference
with torch.no_grad():
    outputs_classifier = model_classifier.generate(input_ids=input_ids, max_new_tokens = 10)
    outputs_classifier_text = tokenizer.batch_decode(outputs_classifier.detach().cpu().numpy(), skip_special_tokens=True)
    eval_preds.extend(outputs_classifier_text)

correct = 0
total = 0
for pred, true in zip(eval_preds, dataset_validation[0]["category"]):
    print(pred)
    print(true)
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy} % on the validation dataset from classification")