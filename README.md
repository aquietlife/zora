# Zora

Zora is an automatic speech recognition (ASR) library and platform focused on interpretability, openness, and personalization.

## Values

- Open-source: All of the code is available to use on this Github page

- Open weights: All of the model weights are also open and available on the Zora web platform

- Open implementation: All models implementations are provided and are made to be as transparent as possible

- Open data: When needed, all data used to train models are free, open-source, and in the public domain

- Interpretable: Zora provides ways to understand what is going on inside of your model, using techniques like feature visualization and mechanistic interpretability. No black boxes!

- Personal: Zora is made for you. It isn't optimized for all, and instead provides ways to made ASR models that are specific, not general. It's not for all, but for those that need.

- Local-first: Everything is run on your own computer, and nothing is sent to a remote server / "the cloud

- Community-based training: Current model weights were trained on Heap, a community computer cluster at Recurse Center

- Approachable: Zora is a small library that is meant to be easily understood and modified by you

- Transparent

- Small

- Metrics and Evaluations

# Models

Zora comes with two models:

## Personal ASR

- Audrey: A CNN-based speech recognizer. A current implementation shows you how to make an ASR system that recognizes digits (0-9). This model also has interpretability functionality like feature visualization.

# General ASR
- Speech-Transformer: A transformer-based speech recognizer that uses public domain, open datasets for general purpose speech recognition. You are also able to fine-tune the model to make it better at hearing specific voices you want it to recognize. This model also has interpretability functionality through mechanistic interpretability techniques.

# Metrics and Evaluation
- Facilitated through Weights and Biases

# Zora Platform

- For hosting training data and weights

# Open Training

- On Heap, show specs of machine it was trained on

# How to Use

## Requirements

- numpy

- pre-commit

# Development

Create a virtual environment

`conda create -n zoraspeech`

Activate the environment

`conda activate zoraspeech`

Install the package in editable mode

`pip install -e .`

Load up `notebooks/develoment.ipynb` for examples on how to develop the library

Install pre-commit hooks:

`pre-commit install`

## Installation

`pip install zoraspeech`

## Usage

`from zoraspeech import Listener, Architecture, Weights, Interpreter, Evals`


`listener = listener(model_architecture: architecture, model_weights: weights) # model_architecture and model_weights can also be loaded by a string. model_weights are loaded in locally for now but would be retreived from a community-based data trust eventually` 


`listener.listen(audio_buffer)`


`listener.interpret(interpreter) # defined by model architecture, must be registered somehow`