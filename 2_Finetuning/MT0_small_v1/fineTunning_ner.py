from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import random
import wandb

device = "cuda"
model_name_or_path = "bigscience/mt0-small"
tokenizer_name_or_path = "bigscience/mt0-small"
dataset_name = "nelson2424/Grocery_chatbot_text_v1"
checkpoint_name = "grocery_ner_v1.pt"
text_column = "text"
label_column = "items"
max_length_input = 64
max_length_output = 64
lr = 1e-3
num_epochs = 10
batch_size = 6

ner_prompt = """
                ### Instruction:
                Identify the groceries in the following text.
                ### Task
                Text: {text}
                items:
"""

# creating model
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.03)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.temperature = 0.8
model.config.top_k = 20


wandb.init(
    # set the wandb project where this run will be logged
    project="Groceries_ner",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "architecture": model_name_or_path,
        "dataset": dataset_name,
        "epochs": num_epochs,
        "temperature":model.config.temperature,
        "top_k":model.config.top_k,
        "max_length_input":max_length_input,
        "max_length_output":max_length_output,
        "batch_size": batch_size,
        
    }
)

wandb.run.tags = ["no_prompt"]

dataset_train,dataset_test,dataset_validation = load_dataset(dataset_name,split=['train[:80%]', 'train[80%:90%]','train[90%:100%]'])

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def preprocess_function(examples):
    # inputs = [ner_prompt.format(text=x) for x in examples[text_column]]
    inputs = examples[text_column] 
    targets = [ x+". " for x in examples[label_column]]
    model_inputs = tokenizer(inputs, max_length=max_length_input, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=max_length_output, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_dataset_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset_train.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

processed_dataset_test = dataset_test.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset_test.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)


train_dataloader = DataLoader(processed_dataset_train, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

eval_dataloader = DataLoader(processed_dataset_test, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"epoch={epoch}: train_ppl={train_ppl} train_epoch_loss={train_epoch_loss} eval_ppl={eval_ppl} eval_epoch_loss={eval_epoch_loss}")
    wandb.log({"train_ppl": train_ppl, "train_epoch_loss": train_epoch_loss, "eval_ppl":eval_ppl, "eval_epoch_loss":eval_epoch_loss})

correct = 0
total = 0

eval_preds = [x.split(".")[0] for x in eval_preds]

for pred, true in zip(eval_preds, dataset_test[label_column]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy} % on the evaluation dataset")

wandb.log({"accuracy":accuracy})
          
for i in range(0,10):
    print(f"Text: {dataset_test[text_column][i]}\n")
    print(f"Prediction: {eval_preds[i]}\n")
    print(f"Ground Truth: {dataset_test[label_column][i]}\n\n\n")

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

model.push_to_hub("nelson2424/mt0-small-lora-finetune-grocery-ner")

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

model.eval()
i = random.randint(0,15)
inputs = tokenizer(dataset_test['text'][i], return_tensors="pt")
print(dataset_test['text'][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_length_output)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))