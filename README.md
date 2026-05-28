# MSAI 495 Generative AI - Text Generation Project

## Overall Architecture
The objective of this project is to create a generative AI model that generates wine reviews. I accomplish this using a decoder-only transformer architecture, a BPE tokenizer, and a manually implemented attention mechanism. Upon completion of training my model, I incorporate it into a chatbot GUI via Streamlit for users to interact with.

### Dataset
The dataset used is the Kaggle Wine Reviews Dataset ([https://www.kaggle.com/datasets/zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)). This contains 130k rows of wine reviews, including various characteristics. The dataset is first formatted to create a corpus of text that can be used to train the model. The corpus is created by concatenating the 'variety', 'vintage', and 'description' columns of the dataset.

### Tokenization
While my project proposal initially stated I would perform character level encoding, I opted to switch to byte-pair encoding (BPE) based on feedback. To accomplish BPE, I utilized the HuggingFace Tokenizer and BPE libraries. I start by pre-tokenizing the corpus into bytes. I then utilize the BPE algorithm on the corpus to learn the most common byte pairs. The tokenizer is trained to have a vocabulary size of 5,000 tokens. The final tokenizer is saved to a JSON file and can be used to encode and decode text.

### Attention Mechanism

### Transformer

### Training Loop/Hyperparameters

### UI
