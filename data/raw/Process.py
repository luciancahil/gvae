f = open("Train.csv", "r")

char_int_dict = dict()

num_chars = 0
max_size = 0
line_num = 0
threshold = 100
count = 0
for i, line in enumerate(f):
    max_size = max(max_size, len(line))

    if(len(line) < threshold):
        count += 1

    for c in line:
        if c not in char_int_dict and c != '\n':
            char_int_dict[c] = num_chars
            num_chars += 1
    
    if(i % 500 == 0):
        print(i)


print(char_int_dict)
print("Largest: " + str(max_size))
print("Count: " + str(count))
print(float(count) / 71634)
