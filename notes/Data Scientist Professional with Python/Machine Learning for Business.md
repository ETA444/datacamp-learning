
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

