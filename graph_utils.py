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
