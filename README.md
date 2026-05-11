EPITECH PROJECT, 2026 - TARDIS

tardis_model.ipynb
This notebook focuses on building and evaluating regression models in order to predict the "Average delay of all trains at arrival".

Data Loading:
    The notebook starts by importing the necessary libraries (pandas, numpy, scikit-learn) and loading the cleaned dataset.

Feature selection:
    We selected variables known before the journey starts, such as the Service type, Departure/Arrival stations, Month, Year and the Number of scheduled trains. For a predictive model to be useful in the real world, it must only use information that is available at the moment the prediction is requested. If we included columns like "Number of cancelled trains", we would be using data that is only known during or after the journey. Moreover, each feature represents a logical factor in potential delays (e.g. "Departure/Arrival stations" -> certain stations are more congested than others, the route represents a major factor in delay probability; "Month" -> weather condition and holiday travel periods significantly impact train's performance).

Data preprocessing:
    We used One-Hot Encoding (created columns with numerical values of 0/1) to transform text data into a numerical format that the models can understand. Afterwards, we split the dataset: 80% into a Training Set and the remaining 20% into a Test Set to ensure we can evaluate the model on unseen data.

Models:
    1) Baseline Model: A simple strategy that uses the average delay of the training set as a constant prediction for all cases. This provides a minimum performance.
    Baseline Mean Prediction: 5.98 minutes.
    Baseline MAE: 3.19 minutes
    Baseline RMSE: 5.75 minutes.
    MAE (Mean Absolute Error) -> the average of all absolute errors. It shows how many minutes, on average, the model was off by.
    RMSE (Root Mean Squared Error) -> Because of the fact that the errors are squared before being averaged, RMSE gives much more weight to large errors (outliers). If the RMSE value is much higher than the MAE value, it means that the model is making some very large mistakes on certain cases, even if it's generally accurate on most.
    In our example, the RMSE value is nearly double the MAE value. This gives us the information that the dataset contains some trains that have very big delays.
    2) Linear Regression: This model serves as a first step beyond the simple average. It finds a mathematical, linear equation that takes into account all the features in order to predict the final delay.
    Linear Regression MAE: 2.57 minutes
    R2 score: 0.127
    We notice that the model improved the MAE result, reducing the value from 3.19 to 2.57 minutes. This proves that the features chosen, such as "Month" and "Service" have a real impact on delays.
    The R2 score, or the coefficient of determination, measures how well a model fits data, and how well it can predict future outcomes. It ranges from 0 to 1.
    In our case, the R2 score of 0.127 means that the features currently in our dataset account for 12.7% of the variance in arrival delays. It confirms that we take into account important features, however, a large portion of delays is driven by external factors.
    3) Random Forest Regressor: After the Linear Regression, we implemented a Random Forest Regressor. Instead of searching for a single linear equation, this module is an ensemble method which uses 100 independent trees to analyze the data. Each tree learns different patterns, and their result are averaged to produce the final prediction.
    Random Forest MAE: 2.28 minutes
    R2 score: 0.041
    The Random Forest achieved our best error rate, reducing the MAE to 2.28 minutes. This means that for the vast majority of train journeys, this model is the most accurate at predicting the arrival time.
    However, while the MAE improved, the R2 score dropped to 0.041. This highlights a characteristic of the Random Forest Regressor: it is great at identifying non-linear patterns, but is sensitive to outliers. Because the dataset contains big delays, as shown by the high baseline RMSE, the R2 score is lower.
