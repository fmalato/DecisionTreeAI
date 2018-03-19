import pandas as pd
import numpy as np
import math

attributes = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
              "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
trainingSet = pd.read_csv('/home/federico/Scrivania/Intelligenza Artificiale/Data Sets/winequality-white.csv', sep=';',
                          names=attributes, skiprows=1)

print trainingSet

