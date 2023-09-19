from datasets import load_dataset,Dataset
import jsonlines
import json
import random

def concatenate_random_grocery_list(items):
    num_items = random.randint(1, 15)  # Randomly select number of items
    selected_items = random.sample(items, num_items)  # Randomly select items

    if len(selected_items) > 1:
        if(random.random()>0.7):
            shopping_list = ', '.join(selected_items[:-1]) + f', {selected_items[-1]}'  # Concatenate only with commas
        else:
            shopping_list = ', '.join(selected_items[:-1]) + f' and {selected_items[-1]}' # Concatenate with commas and 'and'
    else:
        shopping_list = selected_items[0]  # If only one item, no need for commas or 'and'

    return shopping_list.lower()

base_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n
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

base_templates = []
    
with open("base_templates_v1.json", 'r') as json_file:
    base_templates = json.load(json_file)
    #json.dump(base_templates, json_file,indent=True)


grocery_items = []

with open("grocery_items_v1.json", 'r') as json_file:
    grocery_items = json.load(json_file)


dataset = []


for template in base_templates:
    for i in range(0,17):
        groceries_txt,grocery_list = concatenate_random_grocery_list(grocery_items)
        new_text = template["text"].format(groceries = groceries_txt)
        dataset.append({"text":new_text,"category":template["category"],"items":groceries_txt})
    
# Save the dataset to a .jsonl file
with jsonlines.open('grocery_actions_v1.jsonl', mode='w') as writer:
    writer.write_all(dataset)
    
# # Load the dataset from the .jsonl file
dataset = Dataset.from_json('grocery_actions_v1.jsonl')

# # Upload the dataset to Hugging Face
dataset.push_to_hub('nelson2424/Grocery_chatbot_text_classification_v1')