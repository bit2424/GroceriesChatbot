from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
import torch

# Load the fine-tuned model and tokenizer
model_name = "bigscience/mt0-small"
model_peft_name = "nelson2424/mt0-small-lora-finetune-grocery-action-classifier"  # Replace with your fine-tuned model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_peft_name)

# Prepare input text
input_text = "I will buy turkey for dinner"
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

# Perform inference
with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
