f = open("Train.csv", "r")
out = open("Seeds.csv", "w")


line_num = 0
next(f)
for i, line in enumerate(f):
    parts = line.split(",")
    out.write(parts[0] + "\n")
    if(i >= 100):
        break
