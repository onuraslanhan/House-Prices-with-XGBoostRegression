# 🏠 House Prices - Advanced Regression (XGBoost)

A regression project predicting house sale prices from the Kaggle "House Prices - Advanced Regression Techniques" competition dataset, using XGBoost.

---

## 📊 Dataset

The Kaggle House Prices dataset: 1,460 houses with 80+ features (lot size, quality ratings, garage details, neighborhood, etc.) and the target variable `SalePrice`.

## 🛠️ Tech Stack

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🌲 Scikit-Learn (`RandomForestRegressor`, `train_test_split`)
- ⚡ XGBoost (`XGBRegressor`)

## 🔍 Process

1. **📥 Data Loading** — Loaded the training set (1,460 rows × 81 columns).
2. **🧹 Cleaning & Missing Values** — Handled missing data differently by type:
   - Columns with extreme missingness (`PoolQC`, `MiscFeature`, `Alley`, `Fence`) were dropped.
   - Categorical columns where missing meaningfully means "not present" (`GarageType`, `BsmtQual`, etc.) were filled with `"None"`.
   - Numeric columns (`LotFrontage`, `GarageYrBlt`, `MasVnrArea`) were filled with the median.
3. **🔢 Encoding** — Converted remaining categorical columns to numeric with one-hot encoding.
4. **✂️ Train/Test Split** — Split the cleaned data into training and test sets.
5. **🌲 Baseline Model** — Trained a `RandomForestRegressor` as a baseline.
6. **⚡ Improved Model** — Trained an `XGBRegressor` and improved it by:
   - Tuning hyperparameters (`n_estimators`, `learning_rate`, `max_depth`)
   - Applying a **log transformation** to `SalePrice`, since sale prices are right-skewed — a small number of very expensive houses distort the raw distribution. Predicting `log(SalePrice)` gave a noticeably better score.
7. **🎯 Prediction vs. Actual** — Sampled individual test predictions and converted them back to real dollar values with `np.exp()` to sanity-check the model against actual sale prices.

## 📈 Results

| Metric | Score |
|---|---|
| 🌲 Random Forest — R² | ~0.83 |
| 💵 Mean Absolute Error | ~$16,550 |
| ⚡ XGBoost (tuned + log target) — R² | ~0.85 |
| 💵 Mean Absolute Error | ~$14,800 |

## ⚠️ Notes & Learnings

- 🌳 Tree-based models (Random Forest, XGBoost) don't need feature scaling (`StandardScaler`) — unlike distance-based models like Logistic Regression, they split on per-feature thresholds, which are scale-invariant.
- 📉 The log transformation of the target was the single biggest improvement in this project — a common technique for right-skewed regression targets.
- 💡 A single prediction can look far off from the actual price (some houses are just harder to predict), which is why **MAE across the whole test set** is a much more reliable performance measure than eyeballing individual predictions.

## 🏆 Kaggle Submission
This project was submitted to the official Kaggle "House Prices - Advanced Regression Techniques" competition leaderboard.

---

🔗 **Related projects:**
- [🚢 Titanic Survival Prediction](https://github.com/onuraslanhan/titanic-survival-prediction)
- [🧬 Gene Expression Classification](https://github.com/onuraslanhan/Gene-Expression-with-RandomForest)
- [💬 Turkish Sentiment Analysis API](https://github.com/onuraslanhan/sentiment-analysis)
