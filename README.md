# Sentiment Classification on IMBD Movie Reviews

This project implements sentiment classification on movie reviews using two different deep learning approaches:

1. **Recurrent Neural Networks (RNN)**
2. **Transformers (BERT)**

## Project Structure

```
sentiment_classification_rnn_bert
│── sentiment_classification_rnn
│   │── layers.py  # Custom Embedding and LSTM layers
│   │── classifier.py  # RNN classifier combining custom layers
│   │── main.py  # colab notebook preprocess data, train and evaluate the RNN model
│── sentiment_classification_bert.ipynb  # Colab notebook implementing sentiment classification with BERT
│── README.md  # Project documentation
```


## Dataset

The dataset used for training can be found [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).



