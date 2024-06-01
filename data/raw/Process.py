f = open("Train.csv", "r")
f1 = open("Test_670000.csv", 'w')

char_int_dict= dict()
for i, line in enumerate(f):
    f1.write(line)

    if(1 >670000):
        break