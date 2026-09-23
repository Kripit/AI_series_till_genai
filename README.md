# AI Series

This is my personal AI and machine learning learning series.

I am building these notes and experiments while learning the ideas properly, from the mathematics and intuition to code that looks more like something I could use in a real project. I am keeping the explanations, mistakes, experiments, and production notes together instead of hiding the learning process.

## What is inside

The main notebook, [AI_series.ipynb](AI_series.ipynb), currently covers:

- Linear regression from scratch and with scikit-learn
- Logistic regression, sigmoid, cross-entropy, and decision boundaries
- L1, L2, and ElasticNet regularization
- Gradient descent, momentum, RMSProp, Adam, learning-rate schedules, and clipping
- Bias-variance tradeoff, learning curves, cross-validation, and nested cross-validation
- Decision trees, Random Forest, XGBoost, OOB evaluation, and model comparison
- CNNs for image classification using a raw image-folder workflow
- RNNs and LSTMs for sentiment classification
- Transformers, attention, Q/K/V, multi-head attention, positional encoding, and transformer blocks
- Hugging Face tokenization, zero-shot classification, fine-tuning, embeddings, semantic search, FAISS, and the foundations of RAG

## Why I am sharing it

Most beginner resources show the final code without showing how the pieces connect. I am trying to document the connections:

- the same gradient ideas appearing in different models
- why scaling, regularization, validation, and data leakage matter
- how the theory becomes a working implementation
- where a from-scratch example stops being useful and a production library should take over
- what the shapes, metrics, and failure modes actually look like when the code runs

This is still a work in progress. Some cells are designed for Google Colab and may need small path or package changes to run elsewhere.

## What I hope people find useful

I hope this helps someone who is learning machine learning and keeps asking, “what is this line actually doing?” It is also a record of my own progress, so I expect the explanations and code to improve as I learn more.

## More coming soon

More modules, experiments, corrections, and practical AI projects are coming soon.

I am sharing the series as it grows rather than waiting until everything feels finished.

## Current status

This repository is actively being expanded. The notebook is the main learning artifact for now; separate, cleaner project implementations will be added as the series develops.