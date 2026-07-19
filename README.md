# Data Science & Machine Learning Preprocessing Pipeline

This repository documents the complete end-to-end process of Data Preprocessing and Exploratory Data Analysis (EDA). The project is structured into logical parts, moving from initial data understanding to preparing features for machine learning models.

---

##  LAB1: Dataset Exploration
The initial phase of the project focuses on understanding the raw data structure, characteristics, and quality before applying any transformations.
*   **Load Dataset:** Importing raw data into the workspace (e.g., using Pandas).
*   **Display Shape:** Identifying the dataset dimensions (total rows and columns).
*   **Display Data Types:** Checking feature types (numerical vs. categorical).
*   **Display Summary Statistics:** Analyzing central tendencies (mean, median, min, max, std).
*   **Display Missing Values:** Spotting null or empty entries.
*   **Display Duplicate Records:** Detecting redundant rows.
*   **Display Class Distribution:** Inspecting the balance of target class labels.

---

##  LAB2: Data Visualization
Translating the statistical insights from LAB1 into visual graphs to better recognize trends, patterns, and relationships.
*   **Histogram:** Visualizing the frequency distribution and skewness of numerical variables.
*   **Correlation Heatmap:** Mapping out how numerical features relate to one another to guide feature selection.

---

##  Part 3: Data Cleaning
Refining the dataset by addressing the anomalies and inconsistencies discovered during the exploration phase.
*   **Operations Performed:**
    *   *Missing Value Handling:* Imputing or dropping null values.
    *   *Duplicate Removal:* Deleting identical rows to prevent training bias.
    *   *Incorrect Data Correction:* Cleaning typos and logical data anomalies.
    *   *Data Type Conversion:* Casting features into appropriate numerical or object formats.
*   **Statistical Comparison:** Comparing the **Mean** and **Median** before and after cleaning to ensure the core distribution remains intact.

---

##  Part 4: Feature Engineering
Transforming textual and categorical data into numerical formats so they can be processed by machine learning algorithms.
*   **Label Encoding:** Converting ordinal categories into progressive integers where order matters.
*   **One-Hot Encoding:** Converting nominal categories into binary vectors ($0$ or $1$) to avoid implying an incorrect numerical hierarchy.

---
