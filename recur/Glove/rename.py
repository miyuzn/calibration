import os
import re

# Regular expression pattern to match the files
pattern = re.compile(r'P(\d+)_(\d)R\.csv')

# List all files in the current directory
for filename in os.listdir('.'):
    # Check if the file matches the pattern
    match = pattern.match(filename)
    if match:
        i = match.group(1)
        j = match.group(2)
        # Create the new filename
        new_filename = f"{i}.{j}r.csv"
        # Rename the file
        os.rename(filename, new_filename)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming completed.")
