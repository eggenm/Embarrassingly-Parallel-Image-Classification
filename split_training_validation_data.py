import os
from shutil import copy2

i = 0
directory = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\HCSA'
for filename in os.listdir(directory):
    print(filename)
    trainingDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_training_set\\HCSA\\'
    validationDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_validation_set\\HCSA\\'
    if(i%3==0):
        copy2(str(os.path.join(directory,filename)), str(os.path.join(validationDir,filename)))
    else:
        copy2(str(os.path.join(directory,filename)), str(os.path.join(trainingDir,filename)))
    i = i+1

i = 0
directory = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\NA'
for filename in os.listdir(directory):
    print(filename)
    trainingDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_training_set\\NA\\'
    validationDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_validation_set\\NA\\'
    if(i%30==0):
        copy2(str(os.path.join(directory,filename)), str(os.path.join(trainingDir,filename)))
    elif (i % 40 == 0):
        copy2(str(os.path.join(directory,filename)), str(os.path.join(validationDir,filename)))
    i = i+1

i = 0
directory = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\Not_HCSA'
for filename in os.listdir(directory):
    print(filename)
    trainingDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_training_set\\Not_HCSA\\'
    validationDir = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_validation_set\\Not_HCSA\\'
    if(i%4==0):
        copy2(str(os.path.join(directory,filename)), str(os.path.join(trainingDir,filename)))
    elif (i % 5 == 0):
        copy2(str(os.path.join(directory,filename)), str(os.path.join(validationDir,filename)))
    i = i+1