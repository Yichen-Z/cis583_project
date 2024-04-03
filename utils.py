"""
Useful functions to be used in Notebooks and other scripts as needed. 
For now, still retains the image-related functions from the first project idea.
"""
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from urllib.parse import urlparse


# Pandas dataframe helper functions
def get_slimmer_numerics(raw_df: pd.DataFrame, subset_cols: list = None) -> pd.DataFrame:
    """Downcast int64 and float64 columns to save space. Adapted from https://stackoverflow.com/questions/57531388/how-can-i-reduce-the-memory-of-a-pandas-dataframe"""
    cols = subset_cols if subset_cols is not None else raw_df.columns.tolist()
    for col in cols:
        col_type = raw_df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = raw_df[col].min()
            c_max = raw_df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    raw_df[col] = raw_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    raw_df[col] = raw_df[col].astype(np.uint32)
            else:
                if c_min > np.iinfo(np.float32).min and c_max < np.iinfo(np.float32).max:
                    raw_df[col] = raw_df[col].astype(np.float32)
    return raw_df


def load_feather(file_name: str, drop_col: str = 'index') -> pd.DataFrame:
    """Saving pd.DataFrame to feather format may need a redundant 'index' column. This ensures its removal when reading in."""
    df = pd.read_feather(file_name)
    if drop_col in df.columns > 0:
        df.drop(columns=list([drop_col]), inplace=True)
    return df


def save_feather(raw_df: pd.DataFrame, save_path: str):
    """Automatically ensures 'index' column is created to avoid error and proper file extension is added to save_path."""
    raw_df = raw_df.reset_index()
    raw_df.to_feather(f'{save_path}.feather')


# Text Processing
def add_stopwords_missing_apostrophe() -> list:
    stop_words = stopwords.words('english')
    for word in stop_words:
        if "'" in word:
            stop_words.append(re.sub("'", "", word))
    return stop_words

            
def process_text(text_chunk: str, stopwords: set, lemmatizer_obj: WordNetLemmatizer) -> str:
    """Removes everything but alphanumeric characters and stopwords. Lowercases all letters."""
    try:
        # Replace brackets
        sent = re.sub('[][)(]', ' ', text_chunk)

        # remove URLs
        sent = [word for word in sent.split() if not urlparse(word).scheme]
        sent = ' '.join(sent)

        # remove escape characters
        sent = re.sub(r"\@\w+", " ", sent)

        # remove html tags 
        sent = re.sub(re.compile("<.*?>"), ' ', sent)

        # get only characters and numbers from text
        sent = re.sub("[^A-Za-z0-9]", ' ', sent)

        # lowercase all words
        sent = sent.lower()
        
        # Remove extra whitespace between words
        sent = [word.strip() for word in sent.split()]
        sent = ' '.join(sent)

        # word tokenization
        tokens = word_tokenize(sent)
        
        # removing words which are in stopwords
        tokens = [t for t in tokens if t not in stopwords]
        
        # lemmatization
        sent = [lemmatizer_obj.lemmatize(word) for word in tokens]
        sent = ' '.join(sent)
        return sent
    
    except Exception as ex:
        print(sent,"\n")
        print("Error ",ex)


# Plot helpers
def get_confusion_matrix(predictions, test_target, plot_name) -> None:
    """Compares classification predictions to actual target values to get cconfusion matrix heatmap plot."""
    cm = confusion_matrix(predictions, test_target)
    print(cm)

    sns.heatmap(cm, annot=True, cmap='rocket_r', fmt='d')
    plt.title(plot_name)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def get_classification_report(model, features, target, plot_title) -> None:
    """Output confusion matrix visual and classification metrics report."""
    y_pred = model.predict(features)
    get_confusion_matrix(predictions=y_pred, test_target=target, plot_name=plot_title)
    print(classification_report(y_pred, target))


def get_epoch_graphs(history, metric):
    """From TensorFlow tutorial"""
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax