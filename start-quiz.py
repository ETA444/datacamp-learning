import os
import json
import random
import sys

# Path to your JSON file. If no path is provided, it defaults to the current directory.
path = sys.argv[1] if len(sys.argv) > 1 else "."


# Quiz program

def create_quiz(data):
    while data:
        # Choose a random key-value pair and remove it from the data dictionary
        func, desc = data.popitem()
        # Determine the number of incorrect options to create based on the remaining data
        num_incorrect_options = min(len(data), 3)
        # Create the incorrect answer options
        options = random.sample(list(data.values()), num_incorrect_options) + [desc]
        random.shuffle(options)

        print(f"\nWhat does the function '{func}' do?")
        for i, opt in enumerate(options):
            print(f"{chr(65 + i)}. {opt}")

        print("\nTo quit this quiz, type 'Q'.")
        answer = input("\nEnter the letter corresponding to your answer: ").upper()
        while answer not in ['A', 'B', 'C', 'D', 'Q']:
            answer = input("Invalid input. Please enter A, B, C, D or Q: ").upper()

        if answer == 'Q':
            print("You've chosen to quit the quiz.")
            break
        elif options[ord(answer) - 65] == desc:
            print("Correct!\n")
        else:
            print("Incorrect. Let's try another one.\n")


# Load data from JSON file
try:
    with open(os.path.join(path, 'quiz_data.json'), 'r') as f:
        data = json.load(f)
    if data:
        create_quiz(data)
    else:
        print("No data loaded from file. Please check 'quiz_data.json'.")
except FileNotFoundError:
    print(f"The file 'quiz_data.json' does not exist in the directory: {path}.")
except json.JSONDecodeError:
    print("Unable to decode 'quiz_data.json'. The file might be corrupted or misformatted.")