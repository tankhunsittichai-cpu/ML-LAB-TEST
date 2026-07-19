## Part 4: Feature Engineering

This section focuses on **Feature Engineering**, specifically dealing with categorical data encoding. Since most machine learning algorithms require numerical inputs, categorical variables must be transformed into a numerical format.

### Encoding Techniques Performed:

*   **Label Encoding:** Converts each unique category into a distinct integer (e.g., Low $\rightarrow$ 0, Medium $\rightarrow$ 1, High $\rightarrow$ 2). This technique is typically used for ordinal data where the categories have an inherent order or ranking.
*   **One-Hot Encoding:** Creates a new binary column (0 or 1) for each unique category in a feature. This approach is ideal for nominal data where no natural ordering exists (e.g., Color: Red, Blue, Green), preventing the model from assuming any unintended numerical hierarchy.

---