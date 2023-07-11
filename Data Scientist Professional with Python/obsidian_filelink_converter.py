import glob
import re

# Get a list of all .md files in the current directory
files = glob.glob("*.md")

for filename in files:
    with open(filename, 'r') as file:
        filedata = file.read()

    # Find all obsidian-style links and replace them with standard markdown links
    filedata = re.sub(r'!\[\[Pasted image (.*?)\.png\]\]', r'![Pasted image \1](/images/Pasted%20image%20\1.png)', filedata)

    with open(filename, 'w') as file:
        file.write(filedata)