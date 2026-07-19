## Part 3: Data Cleaning

This section covers the **Data Cleaning** process, which is a crucial step to ensure data quality and integrity before feeding the data into machine learning models. It involves fixing corrupted, incorrectly formatted, duplicate, or incomplete data within the dataset.

### 1. Operations Performed (Perform):

*   **Missing Value Handling:** Dealing with incomplete data by either dropping missing fields or imputing them with appropriate statistical values (such as mean, median, or mode).
*   **Duplicate Removal:** Identifying and removing duplicate rows to prevent bias and ensure that each observation is unique.
*   **Incorrect Data Correction:** Fixing anomalies, typos, or logically incorrect values within the dataset to maintain consistency.
*   **Data Type Conversion:** Converting columns to their appropriate data types (e.g., changing numerical values stored as strings into actual integers or floats) for accurate computations.

### 2. Statistical Comparison (Compare):

After cleaning the data, a before-and-after comparison is conducted to observe how the dataset's central tendency has shifted:

*   **Mean:** Comparing the average values before and after data cleaning (especially after handling missing values) to see if the overall average was impacted.
*   **Median:** Comparing the middle values of the distribution to evaluate the changes and ensure that the cleaning process did not inadvertently distort the data's core structure.

---