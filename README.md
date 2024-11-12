# Sentiment Analysis using Deep Learning Models

This project explores various deep learning approaches for sentiment analysis on Twitter data, focusing on identifying positive, negative, and neutral sentiments in tweets. The project compares the performance of Long Short-Term Memory (LSTM) models, Attention mechanisms, Gated Recurrent Units (GRU), and Transfer Learning techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models and Approaches](#models-and-approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Overview
This project demonstrates how different deep learning models can be applied to classify sentiment in tweets, comparing their performance to determine which architecture is most effective. The notebook contains detailed steps, from data preprocessing and model training to evaluation and visualization.

## Dataset
The project uses the **Twitter Airline Sentiment** dataset, available on Kaggle. This dataset contains tweets categorized into three sentiments: positive, negative, and neutral. Additionally, a secondary dataset is used for transfer learning to test model generalization.

**Download Links:**
- [Twitter Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- [Apple Twitter Sentiment](https://www.kaggle.com/seriousran/appletwittersentimenttexts)

## Project Structure
The project is organized as follows:
- **data/**: Directory for storing datasets
- **notebooks/**: Jupyter Notebooks, including:
- **Enhanced_Sentiment_Analysis.ipynb**: Main notebook with detailed comments and explanations
- **models/**: Pretrained models saved for transfer learning
- **README.md**: Project documentation
- **requirements.txt**: Python dependencies

## Models and Approaches
This project implements and evaluates multiple deep learning models:
1. **LSTM Models**: With and without text cleaning
2. **Attention Mechanisms**: For capturing important words in a tweet
3. **GRU Model**: An alternative to LSTM with simpler architecture
4. **Transfer Learning**: Fine-tuning models on a different dataset for improved generalization

Each model is trained, evaluated, and visualized to compare performance metrics such as accuracy and loss.

## Usage
1. **Data Preprocessing**: The notebook contains functions for data cleaning and preprocessing, including tokenization, stemming, and handling special characters.
2. **Training and Evaluation**: Select a model architecture to train on the dataset. Evaluate and visualize its performance.
3. **Transfer Learning**: Use pretrained models on the secondary dataset to explore model generalization.
4. **Results Visualization**: View loss, accuracy, confusion matrices, and word clouds for each model to compare performance.

## Results
- **Best Models**: LSTM and Attention models performed the best, capturing the sequential dependencies in the data effectively.
- **Transfer Learning**: Models pretrained on the Twitter Airline Sentiment dataset generalized well to the Apple Twitter Sentiment dataset, indicating their robustness.

## Future Work
Potential extensions of this project:
- Exploring **Bidirectional LSTM** or **transformers** for improved performance
- Expanding the dataset for broader sentiment categories
- Investigating additional regularization techniques to enhance generalization

## Acknowledgments
Special thanks to Kaggle for the datasets and the contributors to open-source libraries like TensorFlow and Keras, which made this project possible.
