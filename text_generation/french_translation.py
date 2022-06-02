import pandas as pd

data = pd.read_csv("D:\\archive\\en-fr.csv", nrows=1000000)
targ = data['en'].to_list()
inp = data['fr'].to_list()
del data