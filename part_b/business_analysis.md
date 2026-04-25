# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:** `items_sold` (number of items sold per store per month)

**Input Features:**
- Store features: store_id, store_size, location_type, monthly_footfall, competition_density
- Promotion features: promotion_type
- Calendar features: month, is_weekend, is_festival
- Customer demographics: age_group, average_income

**Type of ML Problem:** This is a supervised regression problem. The model predicts a continuous numerical output (items sold) based on store and promotion features. Regression is chosen over classification because the target is a quantity, not a category.

### B1(b) — Why Items Sold is a Better Target Variable

Revenue is influenced by both price and volume. A promotion like a Flat Discount reduces the price, so revenue may fall even if many more items are sold. Using revenue as the target would penalise promotions that drive high volume but at lower prices — which is misleading.

Items sold (sales volume) directly measures customer response to a promotion, independent of pricing effects. This illustrates a broader principle in ML: the target variable should directly reflect the business outcome you want to optimise, not a proxy metric that is influenced by confounding factors outside the model's control.

### B1(c) — Alternative Modelling Strategy

Instead of one global model, a hierarchical or segmented modelling strategy should be used. Stores can be grouped by location type (urban, semi-urban, rural) and separate models trained for each segment. Alternatively, store_id can be included as a feature with store-level embeddings to capture individual store behaviour. This accounts for the fact that rural stores may respond better to BOGO while urban stores respond better to Loyalty Points — patterns a single global model would average out and miss.

## B2. Data and EDA Strategy

### B2(a) — Joining the Tables

The four tables should be joined as follows:
- **Transactions** is the base table (one row per transaction)
- Join **store attributes** on `store_id` to add store_size, location_type, footfall, competition_density
- Join **promotion details** on `promotion_id` or `promotion_type` to add promotion characteristics
- Join **calendar** on `transaction_date` to add is_weekend and is_festival flags

The grain of the final modelling dataset should be one row = one store per month, with items_sold aggregated as the sum of items sold in that store that month under a given promotion.

### B2(b) — EDA Strategy

1. **Distribution of items_sold by promotion_type** (boxplot) — to see which promotions drive higher sales volume and identify any outliers per promotion.

2. **Items sold by location_type and promotion_type** (grouped bar chart) — to check if certain promotions work better in urban vs rural stores, which would inform the segmented modelling strategy.

3. **Monthly sales trend over time** (line chart) — to identify seasonality and long-term trends, which would influence whether we include time-based features.

4. **Correlation heatmap of numerical features** — to check for multicollinearity between footfall, competition_density, and items_sold, guiding feature selection.

### B2(c) — Handling Promotion Imbalance

If 80% of transactions have no promotion, the model will be biased towards predicting outcomes for no-promotion scenarios and may underestimate the effect of promotions. To address this: use stratified sampling to ensure promotion types are represented in both train and test sets; apply sample weights to upweight promoted transactions; and separately analyse promoted vs non-promoted transactions during EDA to understand baseline vs uplift behaviour.

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Metrics

With 3 years of monthly store-level data, a temporal split should be used — train on the first 2.5 years and test on the last 6 months. A random split is inappropriate because it would allow the model to train on future months and test on past months, causing data leakage and overly optimistic evaluation.

**Evaluation Metrics:**
- **RMSE** — penalises large errors heavily; useful for catching cases where the model badly mispredicts a store's sales
- **MAE** — average absolute error in items sold; directly interpretable to the business (e.g. "on average we are off by 30 items")
- **R²** — proportion of variance explained; indicates overall model fit

### B3(b) — Explaining Different Recommendations Using Feature Importance

To investigate why the model recommends Loyalty Points Bonus in December but Flat Discount in March for Store 12, we would examine the feature importances and the input feature values for both months. In December, features like `is_festival=1` and `month=12` are active — the model has learned that during festive periods, Loyalty Points drive higher sales (customers are planning repeat purchases). In March, these festival flags are absent and `competition_density` may be higher — the model recommends Flat Discount as a competitive response. We would present this to the marketing team using a SHAP (SHapley Additive exPlanations) plot showing which features pushed the prediction towards each promotion.

### B3(c) — Deployment Process

**Saving the model:** Serialise the trained pipeline using `joblib.dump()` to save both the preprocessor and model in a single file.

**Monthly prediction process:** At the start of each month, collect the new month's store attributes and calendar features for all 50 stores. Run these through the same preprocessing pipeline and call `pipeline.predict()` to generate promotion recommendations for each store.

**Monitoring for degradation:** Track RMSE and MAE monthly on actual vs predicted items sold. Set alert thresholds (e.g. if RMSE increases by more than 20% over 3 consecutive months, trigger retraining). Also monitor for data drift — if the distribution of input features (e.g. footfall, competition_density) shifts significantly, the model may need retraining even before performance degrades.