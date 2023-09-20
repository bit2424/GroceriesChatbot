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

device = "cuda"
model_name_or_path = "bigscience/mt0-small"
tokenizer_name_or_path = "bigscience/mt0-small"

checkpoint_name = "grocery_action_classifier_v1.pt"
text_column = "text"
label_column = "category"
max_length_input = 256
max_length_output = 7
lr = 1e-3
num_epochs = 5
batch_size = 8

# creating model
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

text_classification_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n
                ### Instruction:
                
                Classify the text as one of the following categories:
                 -Add elements to the list
                 -Remove elements from the list
                 
                If the text doesn't fit any categories, classify it as the following:
                 -Not a valid command
                Return a JSON object that has the fields category and explanation.
                
                ### Task
                Text: {text}
                Category: {category}
"""

dataset_train,dataset_test,dataset_validation = load_dataset("nelson2424/Grocery_chatbot_text_classification_v1",split=['train[:80%]', 'train[80%:90%]','train[90%:100%]'])


# dataset_train = dataset["train"].train_test_split(test_size=0.15)

# classes = dataset["train"].features["label"].names
# dataset = dataset.map(
#     lambda x: {"text_label": [classes[label] for label in x["label"]]},
#     batched=True,
#     num_proc=1,
# )

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
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


train_dataloader = DataLoader(
    processed_dataset_train, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
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
    print(f"epoch={epoch}: train_ppl={train_ppl} train_epach_loss={train_epoch_loss} eval_ppl={eval_ppl} eval_epoch_loss={eval_epoch_loss}")

correct = 0
total = 0
for pred, true in zip(eval_preds, dataset_test["category"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy} % on the evaluation dataset")
print(f"{eval_preds[:10]}")
print(f"{dataset_test['category'][:10]}")

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

model.push_to_hub("nelson2424/mt0-small-lora-finetune-grocery-action-classifier")

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

model.eval()
i = 13
inputs = tokenizer(dataset_test['text'][i], return_tensors="pt")
print(dataset_test['text'][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))