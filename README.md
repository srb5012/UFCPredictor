# UFC Fight Prediction Models

This repository contains two machine learning projects for predicting the outcomes of UFC fights. The models are trained on the ["Ultimate UFC Dataset"](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset/data) from Kaggle, using a combination of base statistics, advanced engineered features, and (in the second model) betting market odds.

Two distinct models are developed:
1.  **`ufcFightPredictor.ipynb`**: A model that predicts fight outcomes based purely on fighter and fight statistics, without any knowledge of betting odds.
2.  **`ufcFightPredictorWithOdds.ipynb`**: A more powerful "market-aware" model that incorporates betting odds as predictive features, demonstrating their significant impact on predictive accuracy.

## Key Features

The models leverage a rich set of features, listed below:

-   **Base Fight Features**: Fundamental, pre-calculated features from the original dataset describing the fight's context and basic competitor differences.
    -   `TitleBout`: Whether the fight is for a championship title.
    -   `NumberOfRounds`: The scheduled number of rounds.
    -   `AgeDif`, `HeightDif`, `ReachDif`: Physical differences between fighters.
    -   `WinStreakDif`: Difference in the fighters' current winning streaks.

-   **Advanced Engineered Features (EWMA-based)**: Sophisticated features calculated to capture a fighter's recent form and style, giving more weight to recent fights.
    -   `ewma_sig_str_dif`: Difference in recency-weighted significant strikes landed.
    -   `ewma_td_dif`: Difference in recency-weighted takedowns landed.
    -   `schedule_dif`: Difference in the recency-weighted "strength of schedule," based on opponent ranks.
    -   `style_dif`: A recency-weighted metric quantifying the "striker vs. grappler" matchup.
    -   `finishing_rate_dif`: Difference in the recency-weighted rate of finishing fights (KO/Submission).

-   **Market-Based Features (Odds-Aware Model Only)**: These features represent the betting market's collective prediction and are highly predictive.
    -   `RedOdds` & `BlueOdds`: The American-style betting odds for each fighter.

-   **Categorical Feature Transformation**:
    -   `WeightClass`: One-Hot Encoded to treat each weight class as a distinct category without assuming a numerical order.

## Methodology

The workflow in both notebooks follows these key steps:

1.  **Data Loading & Preparation**: The dataset is loaded and critically sorted by `Date` to prevent data leakage from future fights into the past.
2.  **Advanced Feature Engineering**: A fighter-centric historical view is created using an Exponentially Weighted Moving Average (EWMA) on key performance metrics. This ensures that a fighter's recent performance is weighted more heavily.
3.  **Preprocessing**: The final feature set is prepared, including One-Hot Encoding for categorical data and median imputation for missing values (common for a fighter's debut or in older data).
4.  **Model Training & Tuning**:
    -   **Random Forest** and **XGBoost** classifiers are trained.
    -   `TimeSeriesSplit` is used for cross-validation, which is essential for time-series data to ensure the model always trains on past data to predict future fights.
    -   `RandomizedSearchCV` is employed to efficiently find the optimal hyperparameters for both models.
5.  **Evaluation**: The models are compared based on their cross-validated accuracy, and the feature importances of the winning model are plotted to understand which factors are most influential in its predictions.

## Results & Key Findings

### Model 1: `ufcFightPredictor.ipynb` (No Odds)

This model provides a baseline for what can be predicted using only fighter statistics.

-   **Best Cross-Validated Accuracy**: **~58.4%** (XGBoost)
-   **Key Insight**: The model performs slightly better than a coin flip. The most important predictive features were a mix of basic differentials and the newly engineered EWMA features, particularly `WinStreakDif`, `ewma_sig_str_dif`, `AgeDif`, and `ReachDif`.

### Model 2: `ufcFightPredictorWithOdds.ipynb` (Odds-Aware)

This model demonstrates the power of including market data.

-   **Best Cross-Validated Accuracy**: **~65.3%** (Random Forest)
-   **Key Insight**: Including betting odds provides a significant boost in accuracy (~7 percentage points). The `RedOdds` and `BlueOdds` features are, by a large margin, the most important predictors. This confirms that betting markets are highly efficient and serve as a strong aggregate predictor of fight outcomes.

# C++ Decision Tree Classifier (ID3)

In order to practice the core concepts of the machine learning algorithms used in this project, I created this simple C++ program. This program is decision tree classifier based on the **ID3 (Iterative Dichotomiser 3)** algorithm. The program reads a dataset from a CSV file, builds a predictive model by recursively splitting the data based on information gain, and then allows for interactive predictions on new data instances.

## Features

-   **Builds a Decision Tree**: Constructs a tree model from a provided CSV dataset.
-   **ID3 Algorithm**: Uses entropy and information gain to find the optimal feature for each split.
-   **Interactive Prediction**: After training, you can input new data instances to get a prediction.
-   **Tree Visualization**: Prints a simple text-based representation of the learned tree structure to the console.
-   **Standard C++**: Written in standard C++ with no external library dependencies.

## Data Format Requirements

The program expects a CSV file with the following specific format:

-   **Header Row**: The first line of the file must be the header, containing feature names.
-   **Delimiter**: Values must be separated by commas (`,`).
-   **Categorical Data**: All data is treated as categorical strings. The algorithm is not designed for numerical data.
-   **No Missing Values**: The dataset should be complete, as the program does not handle missing values.

To test this program, I used the test.csv file in the repository.

---
Author: Shawn Balgobind
