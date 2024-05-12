# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:00 2024

@author: Ícaro de Lima Nogueira
"""

import matplotlib.pyplot as plt


def plot_histogram(data, xlabel, ylabel, title):
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_scatter(x, y, xlabel, ylabel, title):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_accuracy(range_a, range_b, accuracy_scores, roc_auc_scores):
    plt.figure(figsize=(10,6))
    plt.plot(range(range_a, range_b), accuracy_scores, color="yellow", label='Accuracy')
    plt.plot(range(range_a, range_b), roc_auc_scores, color="blue", label='ROC AUC')
    plt.title('Accuracy and ROC AUC vs. Max Depth')
    plt.xlabel('MAx Depth')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()