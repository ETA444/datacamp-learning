import os
import re

# Walk through each directory and subdirectory in the path
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        # If the file ends with .md
        if filename.endswith('.md'):
            full_path = os.path.join(dirpath, filename)

            with open(full_path, 'r') as file:
                filedata = file.read()

            # Find all obsidian-style links and replace them with standard markdown links
            filedata = re.sub(r'!\[\[Pasted image (.*?)\.png\]\]', r'![Pasted image \1](/images/Pasted%20image%20\1.png)', filedata)

            with open(full_path, 'w') as file:
                file.write(filedata)