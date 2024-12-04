# Chatgpt-GPT4
# Chatbot
Transfer learnt model built on a pretrained LLM such as GPT-2
| **Task**                                     | **Comments**                                                                                          | **Status**      | **Individual Responsible** |
|----------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------|----------------------------|
| **Preprocessing**                            | Handle emojis and punctuations POS tagging,Tokenization, padding, and dataset creation for GPT-2 fine-tuning.                                    | Done            | Ramandeep             |
| **Training**                                 | Three epochs of fine-tuning GPT-2 with proper optimizer.  | Done            | Akash             |
| **Evaluation (ROUGE-L, BERT Scores)**        | ROUGE-L and BERT scores computed for validation set predictions against ground truth responses.       | Done            | Akash            |
| **Interpretation using LIME**                | Placeholder steps for LIME text explanations.                                     | Not applicable         | Akash             |
| **1st round of tuning** | Fine-tuned learning rate from 5e-5 to 1e-4 for better model stability. | Done            | Ramandeep             |
| **2nd round of tuning** | Adjusted training loop for augmented dataset to enhance training diversity.                          | Done            | Akash             |
| **Final AUC Value**             | Achieved AUC value of 1.0 shows Strong distinguishability.                               | Done            | Ramandeep             |
| **Next Steps Recommendations**               | Evaluate on larger datasets, analyze model outputs, experiment with hyperparameters, integrate LIME/SHAP, use user feedback. | Done         | Akash & Ramandeep               |


## Features
- **Fine-tuning**: Leverages the `transformers` library to train the GPT-2 model on the custom conversation dataset.
- **Evaluation**: Computes and displays performance metrics such as ROUGE-L Score, BERT Score, and AUC, and generates ROC curves.
- **Model Interpretability**: Placeholder for integrating interpretability tools for explaining model decisions.
- **Custom Tokenization**: Handles input tokenization with padding and truncation to fit a specific maximum length for efficient model processing.
- **Deployment**: Interactive web app using Streamlit for real-time model testing.

## Data Preparation
The dataset should be in CSV format (`processed_data.csv`) and must have the following columns:
- **`human`**: Contains the human-side of the conversation.
- **`gpt`**: Contains the GPT-generated responses.

0	human	gpt
0	1	ive been feeling so sad and overwhelmed lately...	hey there im here to listen and support you it...
1	2	i recently got a promotion at work which i tho...	i can understand how it can be overwhelming wh...
2	3	well the workload has increased significantly ...	it sounds like youre dealing with a lot of pre...
3	4	ive been trying to prioritize my tasks and del...	its great to hear that youre already implement...
4	5	youre right i havent really opened up about my...	its completely normal to feel that way but rem...

Ensure that the CSV file is placed in the `data/` directory for the scripts to access it.

## Usage Instructions

### Training the Model
The training script fine-tunes a pre-trained GPT-2 model using the provided dataset. The training process involves specifying hyperparameters such as learning rate, batch size, and the number of epochs. This step is crucial for adapting the pre-trained model to the specific nature of the conversation data.

### Evaluating the Model
The evaluation script computes key performance metrics:
- **ROUGE-L Score**: Measures the fluency and overlap of n-grams between generated text and reference responses.
- **BERT Score**: Assesses the semantic similarity between generated responses and ground truths using BERT embeddings.
- **AUC & ROC Curve**: Evaluates the binary classification performance of the model. The AUC score is a measure of how well the model distinguishes between human and GPT responses.
- 
### Output 
Embedding(50258, 768)

Epoch 1/3: 100%|██████████| 13/13 [01:58<00:00,  9.11s/it]
Epoch 1 Loss: 11.1060
Epoch 2/3: 100%|██████████| 13/13 [01:39<00:00,  7.64s/it]
Epoch 2 Loss: 9.6153
Epoch 3/3: 100%|██████████| 13/13 [01:38<00:00,  7.60s/it]
Epoch 3 Loss: 8.3855

### Generated Responses:
'Human: i have a few close friends who have always been there for me maybe its time i lean on them more
GPT: its wonderful that you have a support system already in place remember that its okay to ask for help and lean on those who care about you to help you navigate your way through this difficult time.
'Human: ill definitely make a conscious effort to reach out when i need support it just feels nice knowing that im not alone in this 
GPT: youre never alone in this journey its important to regularly check in with yourself and your emotions are there any selfcare practices you enjoy or would like to explore to help you navigate this conflict.
### Ground Truths:
'Human: i have a few close friends who have always been there for me maybe its time i lean on them more 
GPT: its wonderful that you have a support system already in place remember that its okay to ask for help and lean on those who care about you'
'Human: ill definitely make a conscious effort to reach out when i need support it just feels nice knowing that im not alone in this 
GPT: youre never alone in this journey its important to regularly check in with yourself and your emotions are there any selfcare practices you enjoy or would like to explore to help you navigate this conflict'

ROUGE-L: 0.7967323161792907
BERT Score: [0.838638961315155, 0.8232792615890503, 0.9712232351303101, 0.7971480488777161, 0.8147158026695251, 0.9561942219734192, 0.8509508967399597, 0.8607290983200073, 0.8783012628555298, 0.8457299470901489, 0.9734296798706055, 0.9757768511772156]

### First Round of Tuning
Epoch 1/3: 100%|██████████| 13/13 [02:05<00:00,  9.67s/it]
Epoch 1 Loss: 6.3317
Epoch 2/3: 100%|██████████| 13/13 [01:38<00:00,  7.57s/it]
Epoch 2 Loss: 3.9096
Epoch 3/3: 100%|██████████| 13/13 [01:38<00:00,  7.61s/it]
Epoch 3 Loss: 1.5481

ROUGE-L: 0.7967323161792907
BERT Score: [0.838638961315155, 0.8232792615890503, 0.9712232351303101, 0.7971480488777161, 0.8147158026695251, 0.9561942219734192, 0.8509508967399597, 0.8607290983200073, 0.8783012628555298, 0.8457299470901489, 0.9734296798706055, 0.9757768511772156]

### Second Round of Tuning
Epoch 1/3: 100%|██████████| 14/14 [02:37<00:00, 11.23s/it]
Epoch 1 Loss: 0.5878
Epoch 2/3: 100%|██████████| 14/14 [02:11<00:00,  9.36s/it]
Epoch 2 Loss: 0.4380
Epoch 3/3: 100%|██████████| 14/14 [01:48<00:00,  7.72s/it]
Epoch 3 Loss: 0.3695

ROUGE-L: 0.7772292973259689
BERT Score (F1): [0.8855832815170288, 0.9565747380256653, 0.9999998807907104, 0.9118945598602295, 0.8849402666091919, 0.9001768827438354, 0.7956093549728394, 0.8433223366737366, 0.8383560180664062, 0.7773559093475342, 0.8929117918014526, 0.7789052128791809]

### AUC Score: 1.0000


### Generating Responses
The response generation script allows users to input a prompt and obtain a continuation generated by the fine-tuned GPT-2 model. This helps in testing the model's ability to handle different conversational contexts.

### Deployment with Streamlit
The model is deployed as an interactive web app using Streamlit. This allows users to enter prompts and receive real-time responses from the fine-tuned model. To run the app locally:
1. Save the model: model.save_pretrained("gpt2_finetuned")
tokenizer.save_pretrained("gpt2_finetuned") [ File too big size 474mb] 
2. Created app.py
3. Run the Streamlit app script by executing:
   streamlit run app.py

### Video link:
https://azureloyalistcollege-my.sharepoint.com/:v:/g/personal/akash9_loyalistcollege_com/EcDwkvCHhwdIhTb12xK8e8oBgUjrZrf3Ksh5Oyd7UuLn6Q?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=RfrSh3 



