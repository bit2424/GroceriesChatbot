from datasets import load_dataset,Dataset
import jsonlines
import json
import random

def check_compatibility(item,phrase):
    compatible = False
    
    if phrase == "anything":
        compatible = True
    
    if phrase == "supermarket":
        if item != "Other":
            compatible = True
    
    if phrase == "recipe":
        if item == "Food":
            compatible = True
    
    return compatible

def concatenate_random_grocery_list(items, template):
    num_items = random.randint(1,8)  # Randomly select number of items
    selected_items_full = {}
    shopping_list = []
    
    while(0<num_items):
        possible_item = random.sample(items, 1)  # Randomly select items
        if(check_compatibility(possible_item[0][1],template["type"])):
            if possible_item[0][0] not in selected_items_full: 
                selected_items_full[possible_item[0][0]] = possible_item[0][1] 
                num_items-=1
            
            
    selected_items = [x for x in selected_items_full]
    if len(selected_items) > 1:
        if(random.random()>0.7):   
            shopping_list_str = ', '.join(selected_items[:-1]) + f', {selected_items[-1]}'  # Concatenate only with commas
        else:
            shopping_list_str = ', '.join(selected_items[:-1]) + f' and {selected_items[-1]}' # Concatenate with commas and 'and'
    else:
        shopping_list_str = selected_items[0]  # If only one item, no need for commas or 'and'

    return shopping_list_str.lower(),selected_items_full

def format_grocery_objs(grocery_objs):
    out_str = []
    for k in grocery_objs:
        out_str.append(f"({k}:{groceries_objs[k]})")
    return " ".join(out_str)
    

base_templates = []
    
with open("base_templates_v2.json", 'r') as json_file:
    base_templates = json.load(json_file)
    #json.dump(base_templates, json_file,indent=True)


grocery_items = []

with open("grocery_items_v2.json", 'r') as json_file:
    grocery_items = json.load(json_file)


dataset = []

for template in base_templates:
    for i in range(0,10):
        groceries_txt,groceries_objs = concatenate_random_grocery_list(grocery_items,template)
        if("{groceries}" in template["text"]):
            new_text = template["text"].format(groceries = groceries_txt)
            dataset.append({"text":new_text,"category":template["category"],"items":format_grocery_objs(groceries_objs)+"."})
        else:
            new_text = template["text"]
            dataset.append({"text":new_text,"category":template["category"],"items":"{}."})
            


random.shuffle(dataset)

# Save the dataset to a .jsonl file
with jsonlines.open('grocery_actions_v2.jsonl', mode='w') as writer:
    writer.write_all(dataset)
    
# # Load the dataset from the .jsonl file
dataset = Dataset.from_json('grocery_actions_v2.jsonl')

# # Upload the dataset to Hugging Face
dataset.push_to_hub('nelson2424/Grocery_chatbot_text_v2')