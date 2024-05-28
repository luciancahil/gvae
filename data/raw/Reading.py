import os

# Define the path to the main folder and the output file
main_folder = 'above'
output_file = 'HIV_train_oversampled.csv'
IGNORE_FILE = 'ZINC-downloader-2D-txt.curl'

# Open the output CSV file in write mode
with open(output_file, mode='w', encoding='utf-8') as outfile:
    first_line = "smiles,HIV_active\n"
    outfile.write(first_line)
    # Walk through each folder and file in the main folder
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Open each file and read its lines

            with open(file_path, mode='r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split('	')
                    if (parts[0] == 'smiles'):
                        continue
                    line = parts[0] + ", 0\n"
                    outfile.write(line)
