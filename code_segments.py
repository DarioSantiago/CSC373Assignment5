import json
import gzip

# code to open gzipped json file and read reviews into a list
input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
dataset = []
for l in input_file:
    d = eval(l)
    dataset.append(d)
input_file.close()

# code to split the data
train_data = dataset[:int(len(dataset)*0.8)]
dev_data = dataset[int(len(dataset)*0.8):]

# code to extract dates
dates = []
for i in range(len(dataset)):
    dates.append(int(dataset[i]['date'][:4]))
