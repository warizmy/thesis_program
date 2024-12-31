import pandas as pd

# Load the preprocessed data
data = pd.read_csv('../data/preprocessed/data_preprocessed.csv')

# Count entries that meet the criteria for "Ya"
criteria_met = data[
    (data['Cuti'] >= 2) |
    (data['Total_SKS'] < 115)  
#     (data['IPK'] < 3.6)
]

print("Jumlah entri yang memenuhi kriteria untuk 'Tidak':", len(criteria_met))
print("Contoh entri yang memenuhi kriteria:\n", criteria_met)