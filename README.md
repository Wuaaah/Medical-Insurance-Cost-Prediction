# Medical Insurance Cost Prediction

This project explores the factors that influence medical insurance costs and provides a reliable machine learning model to estimate expected charges based on a person's profile.

## Key Insights from the Data
Through Exploratory Data Analysis (EDA), we uncovered several critical insights that drive insurance costs:
* **Age & Charges:** There is a clear upward trend in costs as age increases. Older individuals tend to incur higher medical charges.
* **The Smoking Penalty:** Smoking is the most significant factor affecting insurance costs. Smokers face vastly higher charges compared to non-smokers, regardless of age.
* **BMI Impact:** High Body Mass Index (BMI) also correlates with increased costs, particularly when compounded with smoking.
* **Demographics:** Other factors like sex, region, and number of children have a smaller, yet observable impact on the overall cost.

## The Process: Why these steps?
We approached this problem in three distinct phases:

1. **Exploratory Data Analysis (EDA)**
   * **Why:** Before building any predictive model, we must understand the data. By visualizing distributions and correlations, we identify which factors actually matter (like smoking and age) and ensure our model isn't learning from noise or anomalies.

2. **Model Training (Ridge Regression)**
   * **Why:** We used a linear regression approach enhanced with Ridge (L2) regularization. Medical costs can be highly sensitive to specific features. Regularization helps us prevent overfitting by penalizing overly complex models, ensuring our predictions remain stable and generalize well to unseen individuals.

3. **API Deployment (FastAPI)**
   * **Why:** A machine learning model is only useful if it can be accessed. By wrapping our model in a fast, modern API, we make it effortlessly consumable for front-end applications, mobile apps, or internal tools.

## How to Use

### 1. Setup the Environment
First, ensure you have installed the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Analysis & Train the Model
If you want to generate the EDA figures and retrain the model to produce `model.pkl`:
```bash
python eda.py
python model.py
```
*Note: The EDA figures will be saved in the `figures/` directory.*

### 3. Start the Prediction API
To launch the FastAPI server:
```bash
uvicorn app:app --reload
```
The server will start on `http://127.0.0.1:8000`.

### 4. Make a Prediction
You can test the API by sending a POST request to `/predict`. For example, using `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 30,
  "sex": "male",
  "bmi": 28.5,
  "children": 0,
  "smoker": "no",
  "region": "southwest"
}'
```
You will receive a JSON response containing the estimated medical charge for that profile!
