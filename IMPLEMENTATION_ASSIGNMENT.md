# TextWave Implementation Assignment

## Prerequisites

Before starting, make sure you have completed the following:

1. **Review lectures** on Retrieval Augmented Generation.  
2. **Study tutorials and resources**:  
   - Skeleton code provided with docstrings for each function.  
3. **Prepare datasets and resources**:  
   - `textwave/qa_resources/question.tsv`: list of reference questions and answers to evaluate your system.  
   - `textwave/storage/*.txt`: containing the knowledge base to evaluate your system.  
   - Skeleton code provided with docstrings for each function. Specifically, review the **`textwave/modules/generator/question_answering.py`** script which leverages the [Mistral API](https://mistral.ai/). You can configure this class by specifying three class arguments:

     - The `api_key` argument should take in your unique API key (string) provided to you once you register for a Mistral account.  
       - Go to [https://mistral.ai/](https://mistral.ai/). Click **“Try the API”** and register for a new account.  
       - Once registered, log in and go to your **Workspace** page.  
       - Click **“API Request” → “Create new key.”** Complete the form and copy your key.  
       - In a terminal, run the command:  
         ```bash
         export MISTRAL_API_KEY=<YOUR_API_KEY>
         ```
         You must run this command each time you open a terminal to run this code. Optionally, add this line to your `~/.bashrc` (Linux or MobaXTerm/Windows) or `~/.zshrc` (macOS).  

       > **NOTE:** For Windows users, you may use Command Prompt (CMD) or PowerShell, but you’ll need to research how to set environment variables.  
       > If you succeed and would like to help others, please share your steps on the Teams channel.

     - The `temperature` parameter controls the randomness of the model’s responses.  
     - The `generator_model` parameter specifies the model (e.g., `mistral-{small|medium|large}-latest`).  

   - Do **NOT** commit these datasets into your GitHub repository.

---

## Required Libraries

- Standard libraries (`os`, `sys`, `math`, `itertools`, etc.)
- `torch`
- `transformers`
- `sentence_transformers`
- `mistralai`
- `faiss-cpu`
- `nltk`
- `qa-metrics`
- `numpy`
- `matplotlib` / `seaborn`
- `pandas`
- `scikit-learn`

> **Note:** The updated `requirements.txt` includes all required packages.

---

## Objectives

By the end of this assignment, you will:

- Implement, extend, and package the **Extraction, Retrieval, and Interface Service**.

---

## Instructions

Use [this GitHub Classroom link](https://classroom.github.com/a/2G4voSFC) to fork the IronClad base repository for this assignment into your personal GitHub account.  
Please update your current repository as there may be new updates.  
Make your changes as directed by the instructions and push your commits to your repository.  
*Unit tests automatically run when you push new commits.*

> **Note:**  
> If you are having access issues, post on the Teams channel **“EN.705.603 - All Sections”** under **“Troubleshooting”**, including your issue and GitHub username.  
> Tag **@Vince Pulido** to expedite the troubleshooting process.

---

### Extraction Service

1. **`preprocessing.py`**

---

### Retrieval Service

1. **`index/*.py`**  
2. **`search.py`**  
3. **`rerank.py`** — You will need to implement **`textwave/modules/utils/bow.py`** and **`textwave/modules/utils/tfidf.py`**.

---

### Interface Service

1. **`app.py`**

---

## Submission & Evaluation

Check in (`git push`) all implementation files into your provisioned GitHub repository.  
Provide the **repository URL** in your Canvas submission before the deadline to receive credit.  
After checking in, **GitHub Classroom** will automatically run the autograder.

1. **Push your work**  
   - Check in (i.e., `git push`) all implementation files into your provisioned GitHub repository.

2. **Submit your repository URL**  
   - Provide the **repository URL** before the deadline to receive credit.

3. **Autograder execution**  
   - Once you push to GitHub, **GitHub Classroom** will automatically run the autograder.  
     - View results under the **Actions** tab in your repository.  
     - Each push triggers a new autograder run.  
     - The autograder runs unit tests and may also check:
       - File naming conventions  
       - Method signatures  
       - Correctness of outputs  
       - Presence of required files  

4. **Checking your grade**  
   - Go to your repository on GitHub.  
   - Click the **Actions** tab.  
   - Select the latest workflow run (triggered by your most recent commit).  
   - Expand the **Autograder job** to see detailed test results.  
   - ✅ indicates a passed test; ❌ indicates a failed test.

5. **Resubmissions**  
   - If you fail tests, fix your code and push again.  
   - Each new commit will re-run the autograder.  
   - Only the **latest successful run before the deadline** counts toward your grade.

---
