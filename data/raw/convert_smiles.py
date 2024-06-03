import selfies as sf

f = open("Train.csv", "r")

symbol_count = dict()
len_count = dict()

char_int_dict= dict()
for i, line in enumerate(f):
    try:
        encoded_selfies = sf.encoder(line.strip())

        symbols = list(sf.split_selfies(encoded_selfies))  # ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_2]']  
    
    except(Exception):
        continue

    length = int(len(symbols) / 90)

    if length in len_count:
        len_count[length] = len_count[length] + 1
    else:
        len_count[length] = 1

    for symbol in symbols:
        if symbol in symbol_count:
            symbol_count[symbol] = symbol_count[symbol] + 1
        else:
            symbol_count[symbol] = 1
    
    if i % 1000 == 0:
        print(i)

print(symbol_count)
print(len_count)