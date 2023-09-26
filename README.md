# Groceries from chat
I want to develop the basic logic to identify the groceries we need or don't need to buy based on chat history, and have a list that keeps track of the actions and elements.

We create a synthetic dataset based and finetune some small LLMs to perform these tasks.

## Requirements

- **Identify the type of query the user is using:**
  - Putting items in a list of needed groceries
    - Identify if the item was already on the list
      - If it was on the list, ask the user whether we want to replace the quantity or if we add up
    - Identify the quantity of the item that is being mentioned
  - Removing items from the list of needed groceries when they are acquired or no longer needed.
  - Print the list of items
  - The list should keep track of the name of an item, the quantity in units, and the weight

## Iteration 1

- The first step is to be able to identify two things in a sentence:
  - Whether we are adding elements or if we are reporting used groceries
  - Identify all the groceries that are being listed in the text
- First, we need to start with a basic Dataset to complete these tasks.
- ### Dataset
    - We are going to use an LLM to create the dataset, and some examples of my family asking for groceries.
    - Interestingly, we can identify the fundamental elements in a sentence and then we could re-structure it to reduce complexity in solving the task and possibly add flexibility in the way the response is handled.

    To format the provided code samples, you can use Markdown code block syntax. Here's the formatted text:

    I'm building a synthetic dataset:
    I have two basic structures:
    A list of possible grocery items

    ```python
    grocery_items = [
        "Apples",
        "Bananas",
        "Oranges",
        "Milk",
        # Add more items as needed
    ]
    ``` 

    A list of pairs of text and the category for that text, now all the text sections have a format variable {groceries},
    that variable will be replaced with random elements of the previous list separated by "," and the final element by an "and".


    ```python
    base_templates = [
        {
            "text": "Let's make sure we have {groceries} on our list.",
            "category": "Add elements to the list"
        },
        # Add more templates as needed
        {
            "text": "I purchased {groceries}, so we can remove them from the list.",
            "category": "Remove elements from the list"
        },
        # Add more templates as needed
        {
            "text": "I've heard that the local market has a special on {groceries}.",
            "category": "Not a valid command"
        },
        # Add more templates as needed
    ]
    ``` 

    The final dataset after combining those elements looks as follows:


    ```python
    {
        "text": "Let's make sure we have candles, english muffins, soap and napkins on our list.",
        "category": "Add elements to the list",
        "items": "candles, english muffins, soap and napkins"
    }

    {
        "text": "Could you add bread, charcoal, frosting, beef, rice, pie filling, rice noodles, eggs, oranges, quinoa, cleaning supplies, frozen pies, pretzels, chia seeds and potatoes to our shopping list, please?",
        "category": "Add elements to the list",
        "items": "bread, charcoal, frosting, beef, rice, pie filling, rice noodles, eggs, oranges, quinoa, cleaning supplies, frozen pies, pretzels, chia seeds and potatoes"
    }

    {
        "text": "We're all set with ramen noodles, cookie dough, umbrella, cleaning supplies, hot sauce, frozen bread and pickles already.",
        "category": "Remove elements from the list",
        "items": "ramen noodles, cookie dough, umbrella, cleaning supplies, hot sauce, frozen bread and pickles"
    }

    {
        "text": "I picked up sugar, english muffins, tortillas, batteries, cashews, broccoli, soda and bulgur earlier. They can be taken off the list now.",
        "category": "Remove elements from the list",
        "items": "sugar, english muffins, tortillas, batteries, cashews, broccoli, soda and bulgur"
    }

    {
        "text": "We should check the reviews for ketchup before buying.",
        "category": "Not a valid command",
        "items": "ketchup"
    }
    ``` 

    **Things to improve in the dataset:**
    - The random selection of items in the grocery list can lead to incongruent results
        - Example:
            - We are buying toothpaste and toilet paper for the recipe 
        - We should use another tag to identify the base template elements, whether they are speaking, food, cleaning supplies, or anything in other category.
        - We should also add another tag to the elements in  grocery items, to determine if they are food, or something else.
        - After that, we can pair up only elements of the base template and the grocery items in a way that makes more sense.
    - We are not doing anything related to the quantities of the elements in the list.
    - We are not taking into account that multiple instructions can be performed on the list, on the same text
    - Need to update the dataset to contain samples where no grocery item is mentioned.
## Training

The general approach I want to explore is how to retrain existing Language Models (preferably small ones that can fit on my GPU) to solve the different tasks I propose. There are different ways to do that:

**Fine-tune a pretrained model**
  - The idea here is to retrain some parts of the Language Model in order to obtain outputs that resemble the expected output.
  - You can retrain the whole model or you can use LoRA (Low Rank), a new technique that allows you to update certain weights of the model in a more efficient way, obtaining better results in less time.
  - You can also use TRL if you want to retrain the model using Reinforcement Learning.

## List Action Classification

Here, we are finetuning different Language Models with less than 500M parameters, to perform text classification. We used LoRA for the finetuning and the model had no problem learning the task. For this task, we used the previously created dataset with the columns "text" and "category".

## NER on Text

Here, we are finetuning different Language Models with less than 500M parameters, to perform Named Entity Recognition. We used LoRA for the finetuning. However, I'm having trouble with some stuttering when my fine-tuned model is predicting things. Some things to improve this problem are:

**To reduce stuttering in the generated responses of your Language Model, you can try a few strategies**:
  - **Temperature and Top-K Sampling**:
    - How to generate text: using different decoding methods for language generation with Transformers.
    - Adjust the temperature parameter when generating text. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused.
    - Use Top-K sampling to limit the set of next-word candidates. This can help avoid repetitive patterns.
  - **Diversify Training Data**:
    - If you have control over the training data, make sure it includes a diverse range of prompts and sentences. This can help the model learn to generate more varied responses.
  - **Experiment with Model Size**:
    - Smaller models might be less prone to generating long, repetitive sequences. You could try using a smaller model and see if it improves the issue. Curiously enough, though it can be counterintuitive.
  - **Train for Fewer Tokens**:
    - You can set a limit on the number of tokens generated by the model to prevent excessively long responses.
  - **Postprocessing**:
    - Improve the prediction by adding an end token, in this case a ".", indicates in a better way when the prediction is finished, and indicates that the rest can be ignored.

So far, I've just been using the MT0 model, but I've been getting some pretty good results in both of them.

## Testing 
- I separated the dataset in three parts: training, testing, and validation.
- In this step, we are using the validation section to put more challenging examples not seen in the dataset before to test the models capabilities. 
- Examples like the following are failling at the current v1 version of the model:
    - category: Not a command
        - We need a ride for the weekend
        - We need to run to get to the party
        - We need to run this weekend
        - Pick up the pace or we will be late
        - We need James to come to the party
        - The {groceries} at that party were awesome
        - The {groceries} at the restaurant were delicious.
    - Pick up towels on the way home  
        - category: Add elements to the list
        - items: towels
- I think I made a mistake, and the model over-learned some stuff
    - Maybe putting some negative examples of actions in the dataset can help.

# Iteration 2

We want to make task more complex to handle the semantics of a phrase better. We want to identify the nature of the identify objects, and used that to improve the prediction of the action that is going to be performed on the list.

##  Dataset
    - We are going to keep the idea of mixing subjects with base phrases, but we are going to add semantics
- Now for each item we have one of the following categories:
    - Food
    - Household Items
    - Health and Beauty Products
    - Decoration
    - Other
    
    It looks as follows:
    ``` python
    items = [
    ["Apples","food"],
    ["Bananas","food"],
    ...
    ["Toilet Paper","Household Items"],
    ["Dish Soap","Household Items"],
    ...
    ["Toothpaste","Health and Beauty Products"],
    ["Soap","Health and Beauty Products"],
    ["Shampoo","Health and Beauty Products"],
    ...
    ["flowers","Decoration"],
    ["ballons","Decoration"],
    ...
    ["Television","Other"],
    ["Computer","Other"],
    ...
    ]
    ```
- For the base templates, we are going to be adding a new attribute called "type" that is going to give us a hint of what type of items can be replaced in the {groceries} variable:
- We have three types:
    - Anything, can be completed with any of the five categories
    - Supermarket, can be completed with all categories except other
    - Recipes, can be completed with only food category items
    
    It looks as follows

    ```python
    [
        {
        "text": "Let's make sure we have {groceries} on our list.",
        "category": "Add elements to the list",
        "type": "anything" 
        },
        {
        "text": "We'll need {groceries} for the gathering.",
        "category": "Add elements to the list",
        "type": "anything"
        },
        ...
        {
        "text": "Remember to include {groceries} in our weekly shopping.",
        "category": "Add elements to the list",
        "type": "supermarket" 
        },
        {
        "text": "Hey, please add {groceries} to our grocery list.",
        "category": "Add elements to the list",
        "type": "supermarket" 
        },
        ...
        {
        "text": "I was thinking of using {groceries} for dinner.",
        "category": "Add elements to the list",
        "type":"recipe"
        },
        {
        "text": "I'm planning to cook with {groceries}.",
        "category": "Add elements to the list",
        "type":"recipe"
        },
    ...
    ]
    ```

## Training
- The training process remains basically the same.- Training

## Validation
- Again, thinking of some challenging examples out of the data distribution, we see the system struggling with examples with only one item that it has not seen in the data before.
- But some improvements identifying phrases that are not a valid command has been made
- 
- 
- For the simple 1 word examples that confuse what the possible shopping item is,, I think a more powerful model can help, a model that can abstract the information of the buying objects better.
- Another thing that can be improved is for the prediction items to be lowercase, just like the writing text elements.
    