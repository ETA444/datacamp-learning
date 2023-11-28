# DataCamp Notes & Quiz

Welcome to my `datacamp-learning` repository! This repository serves as my personal codebook, tracking my journey of leveling up my Python, R, SQL and PowerBI skills on DataCamp\*. It is a compilation of markdown files summarizing various functions, concepts, and lessons I have encountered across different courses. In addition, this repository also hosts a flashcard-style quiz to test your memory on different functions based on the notes.

## Repository Structure

The repository is organized into separate folders for each track. Within these track folders, each course has its own `.md` file. In the main directory, there are two Python scripts - `quizdata_generator.py` and `start-quiz.py`.

Here is a basic view of the repository's structure:

```
datacamp-learning
│   README.md
│   quizdata_generator.py
│   start-quiz.py
│
└───Track_1
│   │   Course_1.md
│   │   Course_2.md
│   │   ...
│
└───Track_2
│   │   Course_1.md
│   │   Course_2.md
│   │   ...
```

## File Structure

Each `.md` file corresponds to a single course and includes the following sections for each function learned in that course:

- **Function Name:** The name of the function along with the library it belongs to. Example: `sns.histplot()`

- **Description:** A brief explanation of what the function does, its context, and its advantages over similar functions in other libraries. 

- **Function Signature:** A representation of the function signature, showing the function's parameters and their default values, if any. This section includes both mandatory and optional parameters.

- **Parameters:** An itemized list detailing each parameter in the function signature. Each item includes the parameter's name, its expected data type, its default value (if any), and a brief description of its role in the function.

- **Example of Use:** One or more practical examples demonstrating how the function can be used. Each example includes the necessary import statements, data preparation (if any), function usage, and results visualization, along with a brief explanation of the example.

- **Additional Information:** This section contains any other useful information about the function. It may include insights, tips, common use cases, potential issues, or any other information that may enhance understanding or usage of the function.

This structure is used consistently throughout the `.md` files, making it easy to follow along and understand the functions and their applications.

## Python Scripts and Flashcard Quiz

There are two Python scripts at the root of this repository:

1. **`quizdata_generator.py`**: This script scans all the markdown files in the repository, extracts the names and descriptions of the functions, and stores them in a JSON file named `quiz_data.json`. This JSON file is used as the source of data for the flashcard quiz.

2. **`start-quiz.py`**: This script initiates a flashcard-style quiz in the console or IDE console based on the data from `quiz_data.json`. It will present a function name and four potential descriptions. One of these descriptions will be the correct one and the rest will be randomly selected incorrect options from other functions. The script allows the user to quit the quiz at any time by typing 'Q'.

To start a quiz, follow these steps:

- First, navigate to the root of the repository in your console or terminal. If you're using GitHub, this would be your repository's main directory.

- Run the `quizdata_generator.py` script. This will generate or update the `quiz_data.json` file with the latest function names and descriptions from your markdown files. Here is the command to run the script:

  ```
  python quizdata_generator.py "path/to/directory/with/mdfiles"
  ```

- Once the JSON file is ready, you can start the quiz by running `start-quiz.py`:

  ```
  python start-quiz.py
  ```

- The script will then prompt you with questions in your console or terminal. Respond by typing the letter of the answer you think is correct.

## Contributing 

While these notes are primarily for my own reference, others who are learning to code might find them useful. If you're one of these people and you'd like to contribute to these notes or suggest improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

##  
##### ~ Thank you for being here ~
##### Sign up to Datacamp for the full experience: [https://www.datacamp.com](https://www.datacamp.com)
##### I use Obsidian to manage and write these notes: [https://www.obsidian.md](https://www.obsidian.md)
##### Add me on LinkedIn: [https://www.linkedin.com/georgedreemer](https://www.linkedin.com/in/georgedreemer/)
---
###### Disclaimer: I did my best to not infringe on Datacamp's rights by mostly only outlining the functions themselves, you can always reach me at `penpal@dreemcorp.com` for modification/removal.
