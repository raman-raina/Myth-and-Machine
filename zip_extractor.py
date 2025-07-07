import zipfile

with zipfile.ZipFile('training_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('training_dataset') 

import zipfile

with zipfile.ZipFile('testing_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('testing_dataset') 