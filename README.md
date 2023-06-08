## Introduction:

A Question Answering system is an artificial intelligence system that provide clear responses to their
inquiries. It analyses user questions using natural language processing, machine learning, and other
cutting-edge technologies before retrieving data from a database or the internet and presenting the
solution in a way that is understandable to humans. By reducing time and effort needed to retrieve
information, these systems employ variety of applications. Question-and-answer systems have evolved
into a crucial tool for effective communication in the modern world, when we have access to a great
amount of information.

## Methodology:
Based on the user's needs, this question-and-answer system will offer a means to extract useful
information from a context. It functions as described in the following steps:
1. To begin with, our system has a question processing component that reads questions and extracts
keywords from them.
2. After retrieving the text input, a corpus processing component searches for the answer in the context.
3. Finally, output is retrieved for the user-posted mandatory query from the context.

## Approach:
The steps that we are going to follow in our project are as mentioned below:
1. Web scrapping the data
2. Creating Custom Dataset and Dataloader
3. Token generation using Regular Expressions
4. Generating Vocabulary
5. Generating Dictionary
6. Creating a Mini Network from Scratch
7. Defining Hyper Parameters
8. Training the model
9. Model Evaluation
10. Transfer Learning & Hugging Face

## Mini Network Architecture
 
The QANet architecture uses a combination of convolutional neural networks (CNNs) and self-attention mechanisms to extract information from the input passage and the question. The CNNs capture local features of the input, while the self-attention mechanisms allow the model to attend to different parts of the input at different levels of granularity.

<p align="center">
<img src="https://github.com/cbass250894/Conversational-Q-A-ChatBot/assets/104287899/3e43cedf-ad91-46f3-a7ec-8839d56604cf" alt="Parameters" width="400" height="400" align="center" />
</p>

QANet is composed of several layers, including embedding layers, convolutional layers, self-attention layers, and output layers. The embedding layers transform the input text into a sequence of vectors, which are then fed into the convolutional layers to extract local features. The self-attention layers are used to compute the global representation of the input by attending to different parts of the sequence. Finally, the output layers predict the start and end positions of the answer within the input sequence.

## Transfer Learning Architecture


<p align="center">
<img src="https://github.com/cbass250894/Conversational-Q-A-ChatBot/assets/104287899/92fdfa64-ca0c-4875-8f92-c30b23e789ee" alt="Parameters" width="400" height="400" align="center" />
</p>

A deep neural network architecture called BERT (Bidirectional Encoder Representations from Transformers) is intended to handle a range of natural language processing (NLP) challenges, including language comprehension, sentiment analysis, and question-answering. Being a bidirectional model, it can comprehend the context of a word depending on both the words that come before and after it in a phrase. This is accomplished via a method known as masked language modeling, where part of the input tokens are arbitrarily hidden, and the model is trained to anticipate the hidden characters based on the input sequence's remaining tokens.
BERT is built on a neural network known as the Transformer architecture, which employs self-attention techniques to record the relationships between various portions of the input sequence. By adding task-specific layers on top of the pre-trained model and then training the entire model on the task-specific dataset, the pre-trained BERT model may be fine-tuned on a particular downstream NLP job. On a number of NLP tasks, including sentiment analysis, named entity identification, and question answering, fine-tuning BERT has demonstrated to achieve state-of-the-art performance.

## Implementation
Refer the uploaded Report

