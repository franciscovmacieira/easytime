# Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official website or the GitHub repository.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
    are welcome :)

## Get Started!

Ready to contribute to `easytime`? Here's how to set up your local development environment.

### 1. Fork and Clone the Repository

First, navigate to the `easytime` GitHub repository and click the "**Fork**" button in the top-right corner. This creates your own copy of the project.

Next, clone your forked repository to your local machine. Replace `[Your-GitHub-Username]` with your actual username.

```console
git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/easytime.git
cd easytime
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to work in a virtual environment. This keeps your project's dependencies isolated.

```console
# Create the virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies in Editable Mode

Install the required packages from the `requirements.txt` file. Then, install `easytime` itself in "editable" mode (`-e .`) so your code changes take effect immediately.

```console
# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```
*(Note: If you don't have a `requirements.txt` file, you can create one in your project with `pip freeze > requirements.txt` after installing all necessary packages.)*

### 4. Create a New Branch

Create a dedicated branch for your changes. Choose a descriptive name.

```console
git checkout -b name-of-your-bugfix-or-feature
```

### 5. Make Your Changes and Run Checks

Now, you can modify the code. When you're done, run the project's tests and code formatters to ensure everything is correct.

```console
# Example: Run tests (if you're using pytest)
pytest

# Example: Format your code (if you're using Black)
black .
```

### 6. Commit Your Changes and Push to Your Fork

Commit your work with a clear message and push the branch to your forked repository on GitHub.

```console
git add .
git commit -m "A clear and descriptive commit message"
git push origin name-of-your-bugfix-or-feature
```

### 7. Open a Pull Request

Finally, go to your fork on GitHub. You should see a prompt to "**Compare & pull request**". Click it, review your changes, and submit the pull request to the main `easytime` repository.

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.
