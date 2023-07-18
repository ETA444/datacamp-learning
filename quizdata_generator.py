import re
import os
import random
import json
import sys

# Description: This generates the .json file with the quiz data. It takes the function names and function
# descriptions from all the markdown files in the given folder, saves them to the .json file, and then you need to use
# the 'start-quiz.py' in your IDE or console to initiate a quiz.

# Path to your Markdown files. If no path is provided, it defaults to the current directory.
path = sys.argv[1] if len(sys.argv) > 1 else "."

markdown_files = []

# Walk through each directory and subdirectory in the path
for dirpath, dirnames, filenames in os.walk(path):
    # For each file in the filenames list
    for filename in filenames:
        # If the file ends with .md
        if filename.endswith('.md'):
            # Append the full file path to the markdown_files list
            markdown_files.append(os.path.join(dirpath, filename))

data = {}

for file in markdown_files:
    with open(file, 'r', encoding='utf-8') as f:  # No need to join with path, as full path is stored in markdown_files
        lines = f.readlines()

    current_function = None
    current_description = None

    for line in lines:
        # Check if the line starts with '## ', indicating it's a function name
        if line.startswith('## '):
            current_function = line[3:].strip()
        # Check if the line starts with 'The `...` function', indicating it's a description
        elif current_function and line.startswith(f"The `{current_function}` function"):
            # Modify the description to start from 'Function...'
            current_description = line.strip().replace(f"The `{current_function}` function", "Function")
            data[current_function] = current_description

# Save data to a JSON file
with open('quiz_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)