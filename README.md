# Zora

Zora is an interpretable machine listening library and platform, focused on voice and speech. The library currently supports automatic speech recognition (ASR), focused on interpretability, openness, and personalization.

[Presentation of Zora, presented at Recurse Center on December 12, 2024](https://docs.google.com/presentation/d/1IAm4o4RvH7SkxXvzgrPFyxN8dIJCRfPEBHzbUFHw3WQ/edit?usp=sharing)

## Values

- Open Source: All of the code is available to use on this Github page

- Open Weights: All of the model weights are also open and available on the Zora web platform

- Open Implementation: All models implementations are provided and are made to be as transparent as possible

- Open Data: When needed, all data used to train models are free, open-source, and in the public domain

- Open Training: Model weights and data contains trainings specs for easy and clear reproducibility

- Interpretable: Zora provides ways to understand what is going on inside of your model, using techniques like feature visualization and mechanistic interpretability. No black boxes!

- Personal: Zora is made for you. It isn't optimized for all, and instead provides ways to made ASR models that are specific, not general. It's not for all, but for those that need.

- Local-First: Everything is run on your own computer, and nothing is sent to a remote server / "the cloud"

- Community-based Training: Current model weights were trained on Heap, a community computer cluster at Recurse Center

- Small, Approachable, and Transparent: Zora is a simple library that is meant to be easily understood and modified by you for experimenting with machine listening models

-  Measurable: Zora comes with support for metrics and evaluations to understand how it performs and what improvements can be made

# Models

Zora comes with one model:

## Personal ASR

- A CNN-based speech recognizer. A current implementation shows you how to make an ASR system that recognizes digits (0-9). This model also has interpretability functionality like feature visualization. The examples demonstrate how to use this model to recreate [Audrey](https://nyuscholars.nyu.edu/ws/portalfiles/portal/576153242/Vocal_Features_From_Voice_Identification.pdf).

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