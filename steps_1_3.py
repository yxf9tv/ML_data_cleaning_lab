import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Dataset 1: College Age Prediction

# Step 1: Question 
# Can we predict the average financial aid amount that a college will provide
# to students based on institutional characteristics?
# Target Variable: aid_value 

# Step 2: Business Metric 
# If we can accurately predict financial aid amounts based on institutional
# characteristics, prospective students can make more informed decisions about
# college affordability and financial planning. As consultants of the students 
# we could make money by saving them money.

# Step 2: Data Preparation

# Load the data
college = pd.read_csv("cc_institution_details.csv")
print(f"College dataset shape: {college.shape}")
college.head()

# Check missing values
print("\nMissing values per column:")
print(college.isnull().sum().sort_values(ascending=False).head(20))

# Correct variable types
categorical_cols = ['state', 'level', 'control', 'basic', 'hbcu', 'flagship']
for col in categorical_cols:
    if col in college.columns:
        college[col] = college[col].astype('category')

print("\nData types after conversion:")
print(college.dtypes)

# Collapse factor levels
# Carnegie Classification - keep top 5, group rest as Other
top_5_carnegie = college['basic'].value_counts().head(5).index.tolist()
college['basic'] = college['basic'].apply(
    lambda x: x if x in top_5_carnegie else 'Other'
).astype('category')

print("\nSimplified Carnegie Classification:")
print(college['basic'].value_counts())

# State - keep top 10, group rest as Other
top_10_states = college['state'].value_counts().head(10).index.tolist()
college['state'] = college['state'].apply(
    lambda x: x if x in top_10_states else 'Other'
).astype('category')

print("\nSimplified State categories:")
print(college['state'].value_counts())

# Drop rows with missing target
print(f"\nMissing values in aid_value: {college['aid_value'].isnull().sum()}")
college = college.dropna(subset=['aid_value'])
print(f"Rows after dropping missing aid_value: {len(college)}")

# Drop unneeded variables
cols_to_drop = ['index', 'unitid', 'chronname', 'city', 'site', 'long_x', 'lat_y',
                'similar', 'nicknames', 'counted_pct', 'cohort_size', 'aid_percentile']
vsa_cols = [col for col in college.columns if col.startswith('vsa_')]
cols_to_drop.extend(vsa_cols)
cols_to_drop = [col for col in cols_to_drop if col in college.columns]
college = college.drop(columns=cols_to_drop)
print(f"\nDropped {len(cols_to_drop)} columns, remaining: {len(college.columns)}")

# Fill remaining missing values with median
numeric_cols = college.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if college[col].isnull().sum() > 0:
        college[col] = college[col].fillna(college[col].median())

print(f"Remaining missing values: {college.isnull().sum().sum()}")

# Examine target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
college['aid_value'].plot.hist(bins=30, edgecolor='black', ax=axes[0])
axes[0].set_xlabel('Average Aid Value ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Financial Aid')
college['aid_value'].plot.box(ax=axes[1])
axes[1].set_ylabel('Average Aid Value ($)')
axes[1].set_title('Financial Aid Boxplot')
plt.tight_layout()
plt.show()

print("\nAid Value Statistics:")
print(college['aid_value'].describe())

# Normalize continuous variables (except target)
numeric_cols_to_scale = [col for col in college.select_dtypes(include=[np.number]).columns
                         if col != 'aid_value']
college[numeric_cols_to_scale] = MinMaxScaler().fit_transform(college[numeric_cols_to_scale])
print(f"\nNormalized {len(numeric_cols_to_scale)} numeric columns")

# One-hot encoding
category_list = list(college.select_dtypes('category').columns)
college_1h = pd.get_dummies(college, columns=category_list, drop_first=False)
print(f"\nShape before encoding: {college.shape}")
print(f"Shape after encoding: {college_1h.shape}")

# Calculate baseline (prevalence of target)
mean_aid = college_1h['aid_value'].mean()
std_aid = college_1h['aid_value'].std()
print(f"\nBaseline - Mean aid value: ${mean_aid:,.2f}")
print(f"Baseline RMSE (predicting mean): ${std_aid:,.2f}")

# Create data partitions (Train, Tune, Test - 70/15/15)
total_rows = len(college_1h)
train_size = int(0.7 * total_rows)
Train_college, Test_college = train_test_split(college_1h, train_size=train_size, random_state=42)
Tune_college, Test_college = train_test_split(Test_college, train_size=0.5, random_state=42)

print(f"\nCollege Aid Data Partitions:")
print(f"  Train: {Train_college.shape}")
print(f"  Tune:  {Tune_college.shape}")
print(f"  Test:  {Test_college.shape}")

# Verify similar distributions across splits
print(f"\nTrain - Mean: ${Train_college['aid_value'].mean():,.2f}, Std: ${Train_college['aid_value'].std():,.2f}")
print(f"Tune  - Mean: ${Tune_college['aid_value'].mean():,.2f}, Std: ${Tune_college['aid_value'].std():,.2f}")
print(f"Test  - Mean: ${Test_college['aid_value'].mean():,.2f}, Std: ${Test_college['aid_value'].std():,.2f}")


# Dataset 2: Job Placement

# Step 1: Question 
# Can we predict the salary that a student will receive upon placement based
# on their academic background and qualifications?
# Target Variable: salary 

# Step 2: Business Metric 
# If we were like consultants or something and can accurately predict salaries based on academic performance and
# background, students can better understand how their academic choices impact
# career outcomes, and institutions can better prepare students for successful
# placements. And we would make money from having the students as clients.

# Step 2: Data Preparation 

# Load the data
placement = pd.read_csv("Placement_Data_Full_Class.csv")
print(f"\n\n{'='*60}")
print("PLACEMENT DATASET")
print(f"{'='*60}")
print(f"Placement dataset shape: {placement.shape}")
placement.head()

# Check missing values
print("\nMissing values per column:")
print(placement.isnull().sum().sort_values(ascending=False))

# Check status column - explains missing salary values
print("\nPlacement Status:")
print(placement['status'].value_counts())

# Correct variable types
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in categorical_cols:
    if col in placement.columns:
        placement[col] = placement[col].astype('category')

print("\nData types after conversion:")
print(placement.dtypes)

# Collapse factor levels - check if needed
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(placement[col].value_counts())
# Note: All categorical variables have few levels, no collapsing needed

# Handle missing values - salary is missing for "Not Placed" students
# Filter to only placed students
print(f"\nOriginal dataset size: {len(placement)}")
placement = placement[placement['status'] == 'Placed'].copy()
print(f"After filtering to placed students: {len(placement)}")
print(f"Missing salary values after filter: {placement['salary'].isnull().sum()}")

# Drop unneeded variables
cols_to_drop = ['sl_no', 'status']
placement = placement.drop(columns=cols_to_drop)
print(f"\nRemaining columns: {placement.columns.tolist()}")

# Check remaining missing values
print(f"Remaining missing values: {placement.isnull().sum().sum()}")

# Examine target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
placement['salary'].plot.hist(bins=30, edgecolor='black', ax=axes[0])
axes[0].set_xlabel('Salary')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Placement Salary')
placement['salary'].plot.box(ax=axes[1])
axes[1].set_ylabel('Salary')
axes[1].set_title('Salary Boxplot')
plt.tight_layout()
plt.show()

print("\nSalary Statistics:")
print(placement['salary'].describe())

# Normalize continuous variables (except target)
numeric_cols_to_scale = [col for col in placement.select_dtypes(include=[np.number]).columns
                         if col != 'salary']
placement[numeric_cols_to_scale] = MinMaxScaler().fit_transform(placement[numeric_cols_to_scale])
print(f"\nNormalized {len(numeric_cols_to_scale)} numeric columns")

# One-hot encoding
category_list = list(placement.select_dtypes('category').columns)
placement_1h = pd.get_dummies(placement, columns=category_list, drop_first=False)
print(f"\nShape before encoding: {placement.shape}")
print(f"Shape after encoding: {placement_1h.shape}")

# Calculate baseline (prevalence of target)
mean_salary = placement_1h['salary'].mean()
std_salary = placement_1h['salary'].std()
print(f"\nBaseline - Mean salary: {mean_salary:,.2f}")
print(f"Baseline RMSE (predicting mean): {std_salary:,.2f}")

# Create data partitions (Train, Tune, Test - 70/15/15)
total_rows = len(placement_1h)
train_size = int(0.7 * total_rows)
Train_placement, Test_placement = train_test_split(placement_1h, train_size=train_size, random_state=42)
Tune_placement, Test_placement = train_test_split(Test_placement, train_size=0.5, random_state=42)

print(f"\nPlacement Data Partitions:")
print(f"  Train: {Train_placement.shape}")
print(f"  Tune:  {Tune_placement.shape}")
print(f"  Test:  {Test_placement.shape}")

# Verify similar distributions across splits
print(f"\nTrain - Mean: {Train_placement['salary'].mean():,.2f}, Std: {Train_placement['salary'].std():,.2f}")
print(f"Tune  - Mean: {Tune_placement['salary'].mean():,.2f}, Std: {Tune_placement['salary'].std():,.2f}")
print(f"Test  - Mean: {Test_placement['salary'].mean():,.2f}, Std: {Test_placement['salary'].std():,.2f}")


# Step 3

print("""
Dataset 1: College Aid Prediction
Strengths:
- Rich feature set with multiple dimensions of college characteristics
- Clear target variable (aid_value) that is continuous and measurable
- Good institutional diversity (public/private, different classifications)

Concerns:
- Many VSA variables had extensive missing values and were dropped,
  which could represent important graduation/retention information
- Multicollinearity likely exists between different percentile measures,
  graduation rates at different time periods, and financial metrics
- Data leakage risk: endowment value may be highly correlated with aid
  since wealthier schools can offer more aid
- Single snapshot in time - aid policies change year to year
- Outliers from small specialized institutions may affect model performance

Can the data address our problem?
Yes - we have a continuous target with good variation, multiple reasonable
predictors, and sufficient sample size. Should consider regularization
(Ridge/Lasso) to handle multicollinearity.

Dataset 2: Job Placement Salary Prediction
Strengths:
- Clear target variable (salary) with good variation
- Comprehensive academic history across multiple education levels
- Relevant features (work experience, specialization, test scores)
- Clean data after filtering to placed students

Concerns:
- Small sample size after filtering to placed students limits model complexity
- Most features are academic performance metrics that are likely correlated
- Multicollinearity between ssc_p, hsc_p, degree_p, mba_p
- Single cohort - patterns may not generalize to different years/job markets
- Selection bias from filtering to only placed students
- No information about industry, company size, or job role

Can the data address our problem?
Yes - suitable for regression with continuous target and relevant features.
Small sample size means we should prefer simpler models or regularization.
Cross-validation will be important given the small dataset.
""")

