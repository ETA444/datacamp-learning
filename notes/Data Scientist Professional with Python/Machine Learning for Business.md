
# Machine learning and data pyramid

## Machine Learning applications
ML is applying statistical or computer science methods on data to:
1. Draw causal insights
	- **What is causing** our customers to cancel their subscription to our services?
2. Make predictions on future events
	- **Which customers** are likely to cancel their subscription next month?
3. Understand patterns in data
	- **Are there groups of customers** who are similar and sue our services in a similar way?

## Data hierarchy of needs
Also called the *Data Needs Pyramid*, it represents a general hierarchy of data needs and their priority.
1. **Collection:** Extract data from source systems. 
2. **Storage:** Store data in reliable storage. 
3. **Preparation:** Organize and clean data to make it usable. 
	- Outlier detection, data quality processes and more.
4. **Analysis:** Understand trends, distributions and segments in the data.
	- These analysis result in dashboards, ad hoc analyses, business score cards and others.
5. **Prototyping & Testing ML:** Interpretable simple models, A/B tests & experiments. 
	- Investigate causal drivers of desired outputs, run experiments to make sure our insights can drive the outputs up.
6. **ML in Production:** Complex models in production, research and automation.
	- Once the organization knows what ML models work it start implementing them in CRM, websites, apps and other methods of production.

# Machine learning principles

## Supervised vs. unsupervised
1. Draw causal insights
	- This is done with **supervised ML**
2. Predict future events
	- This is done with **supervised ML**
3. Understand patterns in data
	- This is done with **unsupervised ML**

### Supervised ML data structure
An example of a data structure for supervised ML is a table with various **input features** and a column with the **target variable** related to said characteristics, like so:
```
|                | Transaction data A | Transaction data B | Transaction data C | Transaction data D | Fraud probability |
|----------------|--------------------|--------------------|--------------------|--------------------|-------------------|
| Transaction 1  |                    |                    |                    |                    |                   |
| Transaction 2  |                    |                    |                    |                    |                   |
| Transaction 3  |                    |                    |                    |                    |                   |
| Transaction ...|                    |                    |                    |                    |                   |
| Transaction N  |                    |                    |                    |                    |                   |

```

Fundamentally a supervised ML model would take the input features and learn rules to predict the target variable on unseen data.

### Unsupervised ML data structure
In contrast to the table above a unsupervised learning data structure would have **input features**, but **no target variable.*

Therefore it would use **input features** to identify groups of data points with similar features to make into segments/clusters.

### ML: Industry examples

#### Marketing
- **Supervised ML:**
	- Predict which customers are likely to purchase next month.
	- Predict each customer's expected lifetime value.
- **Unsupervised ML:**
	- Group customers into segments based on their past purchases.

#### Finance
- **Supervised ML:**
	- Identify key transaction attributes that indicate a potential fraud
	- Predict which customers will default on their mortgage payments.
- **Unsupervised ML:**
	- Group transactions into segments based on their attributes to understand which segments are the most profitable

#### Manufacturing
- **Supervised ML:**
	- Predict which items in production are likely faulty and should be manually inspected
	- Predict which machines are likely to break and need maintenance
- **Unsupervised ML:**
	- Group readings from machine sensors and identify anomalies for potential manufacturing malfunctions

#### Transportation
- **Suprevised ML:**
	- Predict the expected delivery of the parcel
	- Identify the fastest route for driving
	- Predict product demand to prepare enough stock, rent/buy vehicles and hire workers


# Job roles, tools and technologies

When it comes to data jobs we can use the *machine learning and data pyramid* to understand different positions.

1. **Collection:** on this level we have the `infrastructure owners`, who are mostly:
	- Software engineers
	- System engineers
2. **Storage:** here we have `data engineers`. Looking at job titles we have:
	- Database administrators
	- Pipeline engineers
3. **Preparation:** here we have `data engineers` and `data analysts`
4. **Analysis:** here we have `data analysts` and `data scientists`. 
5. **Prototyping & Testing:** here we have `data scientists` and `ML engineers`.
6. **ML in production:** here we have `ML engineers`.

## Team structure

### Centralized structure
All data functions in one central team. 
- Works well for small companies, startups, new organizations. 
- Gets slow once business matures and requires focus.

### Decentralized/Embedded
Each business unit, geography or department have their own data functions. 
- Works well for larger companies.
- Introduces issues with data governance, differences in definitions, redundancies, and added complexity.

### Hybrid
Infrastructure, definitions, methods and tooling are centralized, while application and prototyping decentralized.


# Prediction vs. Inference dilemma


## Inference vs. Prediction models

1. **Inference/Causal models**: The goal is to **understand the drivers of a business outcome**.
	- Inference focused models are interpretable.
	- **Less accurate** than prediction models.
2. **Prediction models:** The prediction itself is the main goal.
	- Are not easily interpretable (work like "black-boxes").
	- Much **more accurate** than inference models.

### Start with the business question
- "What are the main **drivers** of fraud?"
	- **Inference model** is suitable.
- "**How much** conditions X impact ...?"
	- **Inference model** is suitable.
- **Which** transactions are likely fraudulent?
	- **Prediction model** is suitable.
- "Is the patient **at risk** of ...?"
	- **Prediction model** is suitable.


# Inference (causal) models

## What is causality?
- Identify causal relationship of how much certain actions affect an outcome of interest
- Answers the "why?" questions
- Optimizes for model interpretability vs. performance
- Models try to detect patterns in observed data (observational) and draw causal conclusions

## Experiments vs. observations
- Experiments are designed and causal conclusions are guaranteed e.g. in A/B tests
- When experiments are impossible (unethical, too expensive, both) - the models are used(also called observational studies) to calculate effect of certain inputs on desired outcomes
- Experiments are always preferred over observational studies whenever possible

### Best practices
1. Do experiments whenever possible
2. If running experiments all the time is too expensive, run them periodically and use it as benchmark
3. If there is no way to run experiments, build a causal model. This will require an advanced methodology.

# Prediction models

## Supervised vs. Unsupervised learning

- **Supervised models**
	- **Classification**: Predicting **class/type** of an outcome.
		- Churn, cancellation, fraud, purchase, etc.
	- **Regression**: Predicting **quantity** of an outcome.
		- Dollars spent, hours played
- **Unsupervised models**
	- **Clustering:** grouping observations into similar groups or clusters
		- Customer or market segmentation

### Supervised learning types

#### Classification
Target variable is categorical (discrete).
- Will the customer cancel a service?
- Is this transaction fraudelent?
- What is the profession of this user?

#### Regression 
Target variable is continuous (amount of outcome).
- Number of product purchases next month
- Number of gaming hours next year
- Dollars spent on insurance

#### Data collection
Machine learning teams should collect all available data to predict desired outcome with the highest degree of accuracy, e.g. in case of purchase prediction:
- Customer information
- Purchase history, cancellations, order amount
- Browsing history, logs, errors
- Device details and location
- Product/service usage frequency
- ...

### Unsupervised learning types

#### Clustering
Grouping observations into similar groups or clusters
- Customer or market segmentation

#### Anomaly detection
Detecting which observation fall out of the discovered "regular pattern" and use it as an input in supervised learning or a business input.

#### Recommender engines
Recommending products or services to customers based on their similarity to other customers
- Netflix movie recommendation etc.


# Business requirements

## Scoping business needs

1. **What is the business situation?**
   - The company plans to expand to new markets.

2. **What is the business opportunity and how big is it?**
   - Identify the right markets with the biggest demand.

3. **What are the business actions we will take?**
   - Prioritize and invest more in the markets with higher predicted demand.

### Business Scope - Fraud Example

1. **Situation** 
   - The fraud rate has started increasing.

2. **Opportunity**
   - Reduce fraud rate by X%, resulting in Y USD savings.

3. **Action**
   - Work on improving the fraud detection system.
   - Reduce fraud drivers.
   - Manually review transactions at risk.

### Business scope - churn example

1. **Situation**
   - The customers started to churn more.

2. **Opportunity**
   - Reduce churn rate by X%, resulting in Y USD revenue saved.

3. **Action**
   - Work on identifying and improving churn drivers (website errors, too much/little advertising, customer service issues, etc.).
   - Identify customers at risk.
   - Introduce retention campaigns.

## Business situation

### Asking the right questions

Always **start with Inference questions**.
- **Inference Questions:**
	1. **Why has churn started increasing?**
	2. **Which information indicates a potential transaction fraud?**
	3. **How are our most valuable customers different from others?**

Build on inference question to **define prediction questions**.
- **Prediction Questions (Building on Inference):**
	1. **Can we identify customers at risk of churning?**
	2. **Can we flag potentially risky transactions?**
	3. **Can we predict early on which customers are likely to become highly valuable?**


## Business opportunity

### **Opportunity Assessment:**

- **Size up the opportunity:** Determine the potential impact of addressing the identified outcome drivers.
- Once you know the drivers of the outcome, how much will it cost changing them, and what will be the value of doing that?
- Finally, how do you know if you can affect the predicted outcome? (hint - experiments, experiments, and more experiments)


## Actionable machine learning

Finally, how do you know if you can affect the predicted outcome? (hint - experiments, experiments, and more experiments)

1. **First, look at historical levels (churn, fraud, # of high-value customers):** Understand the baseline historical data to gauge the current state.

2. **Run experiments (e.g. target customers at risk with a discount, manually review top 10% riskiest transactions):** Implement experiments to test interventions.

3. **Repeat experiments multiple times, see if you get a repeated pattern of desired results:** Assess the consistency and effectiveness of the interventions.

4. **If yes, use that to calculate opportunity and make a decision if it's a worthwhile investment:** Measure the impact of the experiments to quantify the opportunity and determine if it's worth pursuing.

5. **If no:**
   - **Collect more data:** Gather additional data to gain deeper insights.
   - **Qualitative research:** Conduct qualitative research to understand underlying issues.
   - **Narrow down the business question:** Refine the research question to pinpoint specific areas of improvement.

# Model training

## **Model Training Process**

Machine learning models undergo a structured training process to ensure their effectiveness and generalizability. This process typically involves three critical stages: building the model on the training dataset, assessing intermediate performance on the validation dataset, and finally, measuring the model's final performance on the test dataset.

#### **Building the Model on the Training Dataset**

   During the initial phase, the machine learning model is constructed using a designated training dataset. This dataset consists of labeled examples where the model learns to recognize patterns, relationships, and features that are relevant to the problem it aims to solve. The model adjusts its parameters iteratively to minimize the difference between its predictions and the actual outcomes in the training data. This phase continues until the model reaches a point where it performs well on the training data but has not yet been fine-tuned for generalization.

#### **Intermediate Performance Measured on the Validation Dataset**

   Once the model is trained on the training dataset, it is crucial to assess its intermediate performance using a separate dataset called the validation dataset. This step is essential to prevent overfitting, where the model becomes too specialized to the training data and fails to generalize to unseen data. The validation dataset helps monitor the model's ability to generalize to new, unseen examples. By evaluating its performance on the validation data, we can make adjustments to the model's hyperparameters and architecture, optimizing it for better generalization.

#### **Final Performance Measured on the Test Dataset**

   After refining the model through multiple iterations on the validation dataset, the final performance evaluation is conducted using a separate and untouched dataset known as the test dataset. This dataset serves as a true benchmark of the model's generalization capabilities. By assessing the model on this independent dataset, we can confidently gauge its ability to make accurate predictions on new, real-world data. The test dataset is crucial for estimating how well the model will perform in practical applications and provides a reliable measure of its overall effectiveness.

In summary, the model training process involves a sequential progression from building the model on the training data to fine-tuning it based on intermediate performance on the validation dataset, and ultimately, measuring its final performance on the test dataset. This structured approach ensures that machine learning models not only perform well on familiar data but also maintain their accuracy and reliability when faced with new and unseen information.

## Model performance measurement

### Types
There are two key supervised learning metrics:
1. **Accuracy** which is a metric that tells us how well our model classifies the data. (**classification performance**)
	- Accuracy,  recall, precision
2. **Error** which is a metric that tells us how far our predictions are from the real values. (**regression performance**)

#### Accuracy, Precision and recall
Accuracy looks at overall correctness, precision emphasizes minimizing false positives, and recall emphasizes capturing all true positives. 

These metrics help evaluate different aspects of a model's performance depending on the specific goals and constraints of a task.

- **Accuracy:** Accuracy measures the overall correctness of predictions made by a model. It's the ratio of correctly predicted instances to the total number of instances. High accuracy means the model gets most predictions right but may not be suitable for imbalanced datasets.

- **Precision:** Precision focuses on the accuracy of positive predictions. It measures the ratio of true positive predictions (correctly predicted positives) to all positive predictions made by the model. High precision indicates a low rate of false positive errors.

- **Recall:** Recall, also known as sensitivity or true positive rate, measures the model's ability to capture all actual positive instances. It's the ratio of true positive predictions to all actual positive instances. High recall indicates a low rate of false negatives, ensuring that most positive cases are correctly identified.

##### Optimizing for Precision vs. Recall
The choice of whether to optimize a churn prediction model for precision or recall depends on the cost associated with false positives (Type I errors) and false negatives (Type II errors) and the potential impact of those errors on your business.

1. **Optimizing for Precision:**
   - If the action to prevent churn is costly (e.g., offering expensive customer incentives), you want to ensure that you are targeting the right customers who are truly at risk of churning.
   - In this scenario, you should optimize your model for precision, which means minimizing false positives. A high precision model is conservative in making positive predictions, so the customers it identifies as at risk are more likely to be actual churners.
   - By optimizing for precision, you may reduce the cost associated with offering incentives to customers who are not likely to churn. However, you may miss some potential churners (higher false negatives).

2. **Optimizing for Recall:**
   - If it is very costly to miss a churned customer (e.g., the long-term revenue loss from a churned customer is substantial), you want to ensure that your model captures as many true churners as possible.
   - In this case, you should optimize your model for recall, which means minimizing false negatives. A high recall model is more aggressive in making positive predictions, ensuring that fewer actual churners are missed.
   - By optimizing for recall, you reduce the risk of missing high-value customers who are at risk of churning. However, you may end up offering incentives to some customers who may not have churned (higher false positives).


# Machine learning mistakes

Common mistakes in machine learning are:
- Running machine learning off the bat
- Not enough data
- Target variable definition is wrong
	- What are we predicting?
	- Can we observe it?
		- Contractual churn - customer terminated the premium credit card
		- Non-contractual churn - customer started using another grocery store
- Late testing and no impact
- Feature selection
	- Inference (what **affects** the target variable?)
		- Choose variables that you can control (website latency, price, delivery, etc.)
		- Business has to be involved in feature selection
	- Prediction (can we **estimate** the target variable value in the future?)
		- Start with readily available data
		- Start with a simple model
		- Introduce new features iteratively

# Communication management

## Working groups
Schedule recurring meetings to track progress and define the following:
- Define the business requirements
- Review machine learning model and business products
- Inference vs. prediction
- Baseline model results & outline model updates
- Market testing
- Discuss possible production and implementation

## Business requirements
1. What is the business **situation**?
	- Churn rate has started increasing
2. What is the business **opportunity** and how big is it?
	- Reduce churn from X% to Y%
3. What are the business **actions** we will take?
	- Run retention campaigns targeting customers at risk


## Model performance and improvements
Identify what is the tolerance for model mistakes (remember all models are wrong):
- **Classification**
	- Which class is more expensive to misclassify?
		- Example: it's likely more expensive to mis-classify fraud as non-fraud than vice versa.
- **Regression**
	- What is the error tolerance for prediction?
		- Example: in demand prediction the company will have to buy more inventory than needed if the model error is high


# Machine learning in production

## Production systems
Production system is live, customer facing and business critical.
- Customer Relationship Management (CRM)
	- Example: predicted churn triggers automatic emails
- Fraud detection
	- Example: predicted fraud probability automatically triggers transaction block and requests a manual review
- Online banking platform
	- Example: recommended products shown on the customer's online banking profile
- Autonomous cars
	- Example: autonomous cars use ML in many ways, one example is predicted collision kicks off automatic initiation of brakes and collision avoidance steps.

## Staffing
ML requires various job roles to be made, trained, tested and implemented.

- **Prototype ML**
	- Data Scientists
	- ML Engineers
- **ML in production**
	- Software engineers
	- Data engineers
	- Infrastructure owners
