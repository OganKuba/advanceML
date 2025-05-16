import openml
import pandas as pd

# Pobranie wszystkich datasetów
datasets = openml.datasets.list_datasets(output_format="dataframe")

# Filtrowanie zbiorów:
# - Co najmniej 10 i maksymalnie 50 cech numerycznych
# - Liczba instancji między 2000 a 10 000
filtered_datasets = datasets[
    (datasets['NumberOfNumericFeatures'] >= 10) &
    (datasets['NumberOfNumericFeatures'] <= 50) &
    (datasets['NumberOfInstances'] >= 2000) &
    (datasets['NumberOfInstances'] <= 10000) &
    (datasets['NumberOfClasses'] == 2)
]

# Wybór tylko ID i nazwy datasetów
filtered_datasets = filtered_datasets[['did', 'name']].head(10)

# Wyświetlenie wyników
print(filtered_datasets)
