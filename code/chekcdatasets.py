import openml

datasets = openml.datasets.list_datasets(output_format="dataframe")

filtered_datasets = datasets[
    (datasets['NumberOfNumericFeatures'] >= 10) &
    (datasets['NumberOfNumericFeatures'] <= 50) &
    (datasets['NumberOfInstances'] >= 2000) &
    (datasets['NumberOfInstances'] <= 10000) &
    (datasets['NumberOfClasses'] == 2)
]

filtered_datasets = filtered_datasets[['did', 'name']].head(10)

print(filtered_datasets)
