f = open("HIV_train_oversampled.csv", "r")
out = open("Train.csv", "w")


line_num = 0
next(f)
for i, line in enumerate(f):
    parts = line.split(",")
    out.write(parts[0] + "\n")
    if(i >= 679000):
        break
