# TextWave Analysis and Design Assignment

## Objectives

This assignment will guide you in evaluating the **TextWave System** by:

- Comparing embedding models  
- Analyzing the effects of selecting a particular indexing strategy on retrieval performance  
- Exploring how to optimize the system by selecting configuration parameters  
- Submitting a file (`<BASE_DIRECTORY>/CASE_ANALYSIS.md`) to summarize your findings  
- Submitting supporting Jupyter notebooks (`<BASE_DIRECTORY>/analysis/task*.ipynb`) to show your work  

## Tasks

Please compile all your findings—including every table, figure, and succinct commentary—into a single, cohesive document (`CASE_ANALYSIS.md`).

- Each **table** and **figure** must be clearly labeled (e.g., *Table 1: Model Comparison Metrics*, *Figure 2: PR Curves*).  
- Captions should concisely describe what the reader should observe.  
- After each visualization or table, provide a **brief interpretation** explaining the key takeaway (e.g., *Table 3 shows a 7-point drop in mAP when brightness is reduced by 50%*), and explain how this insight impacts your system design choices.  
- Do **NOT** include step-by-step code.  

Maintain consistent formatting throughout: uniform fonts, spacing, numbering, and captions.  
Ensure the document reads smoothly with explanations that remind the reader **how and why each analysis was performed (methodology)** and **how your results support your overall conclusion**.

You will be evaluated on:

- **Robustness of methodology and reasoning** — Are your conclusions well-supported by evidence, and is your reasoning correct?  
- **Depth and completeness of analysis** — Did you explore trade-offs and alternative explanations?  
- **Clarity and cohesion of presentation** — Does your report read smoothly as a logical narrative?  

For each of the five tasks, include a dedicated Jupyter Notebook (`analysis/task1.ipynb`, `analysis/task2.ipynb`, …, `analysis/task5.ipynb`) showing your analysis, intermediate exploration, and any code used.  
Each notebook should clearly demonstrate how you arrived at your reported results, even if the final report only summarizes them.  
The evaluation will not consider the contents of the notebooks directly—they are supplementary.

---

### Task Breakdown

1. **Chunking Strategy Performance (Text Preprocessing and Index Selection)**  
   Using the list of questions in *textwave/qa_resources/question.tsv*, compare the retrieval performance between the *fixed-length* and *sentence chunking* strategies using a Brute Force indexing approach.  

   For a given question, assume that a chunk is *positive* if it comes from the file listed in the question’s `"ArticleFile"` field (assume each answer lives in exactly one document).  
   Example: Given the question *"Was Abraham Lincoln the sixteenth President of the United States?"*, a chunk is relevant if it is contained within *S08_set3_a4.txt.clean*.  

   Document the parameters used (e.g., chunk size, sentence size, overlap, etc.), and list shortcomings of this evaluation approach when determining preprocessing strategy.  

   Finally, using your chosen chunking strategy:
   - Compare and contrast Brute Force, HNSW, and LSH indexing strategies on retrieval performance.  
   - Compare retrieval performance between implemented re-ranking methods.  
   In both cases, assume that a *relevant* (or *positive*) item is one whose chunk is associated with the document indicated in `"ArticleFile"`.

2. **Generative Model Performance Comparison (Baseline Selection)**  
   Using a "stand-alone" system (i.e., no RAG), develop a baseline by measuring the quality of generated answers.  
   Score answers against **qa_resources/question.tsv**, reporting **overall** and **difficulty-stratified** results (`DifficultyFromAnswerer`: easy/medium/hard).  

   Use the following Mistral models:
   - `mistral-small-latest`  
   - `mistral-medium-latest`  
   - `mistral-large-latest`  

3. **Retrieval-Augmented Generative Model Performance Comparison (Architecture Selection)**  
   Using the optimal strategies from Tasks 1 and 2, compare the generative performance of the Mistral models (small, medium, large) **without the Reranker**.  
   Measure the quality of generated answers against **textwave/qa_storage/question.tsv**.  
   Report both overall and difficulty-stratified results (easy, medium, hard).  
   Compare to the Task 2 baseline and discuss gains or regressions.  

4. **Reranker Performance Comparison (Architecture Selection)**  
   Using the optimal strategies from Tasks 1 and 2, compare the generative performance of Mistral models (small, medium, large) **with rerankers**.  
   Measure the quality of generated answers against **textwave/qa_storage/question.tsv**.  
   Report overall and difficulty-stratified results.  
   Compare these to the baselines from Tasks 2 and 3, discussing gains or regressions.  

5. **Optimize the Number of Retrieved Chunks (Parameter Configuration)**  
   Define `m` as the number of relevant context chunks retrieved for the generative model.  
   Using the optimal model from Task 4, investigate how retrieval performance varies with `m` (e.g., `m = 1, 2, 3, ...`).  
   Suggest the optimal `m`, supported by your findings, and discuss dataset-specific factors that may influence your conclusion.  

---

> **HINT:**  
> For Task 2, modify `textwave/modules/generator/question_answering.py`:
>
> 1. Comment or remove the line in the `generate_answer()` method:
>    ```python
>    f"Context: {', '.join(context)}\n\n"
>    ```
> 2. Change the "system" prompt from:
>    ```python
>    "You must answer the user's questions **only** based "
>    "on the provided context. Do not use any external or prior knowledge. "
>    "Provide clear, concise, and full-sentence answers."
>    "If the context does not mention the answer, respond with 'No context'."
>    ```
>    to:
>    ```python
>    "Provide clear, concise, and full-sentence answers."
>    ```
>
> This modification removes the dependence on contexts.

---

## Submission

- Commit and push your **`CASE_ANALYSIS.md`** file and supporting notebooks (e.g., `analysis/task*.ipynb`) into your provisioned GitHub repository.  
- Provide your **repository URL** when submitting.
