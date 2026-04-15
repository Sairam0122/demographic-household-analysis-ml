import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import sklearn (Library for ML);
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score 

# Load the dataset from Excel file
# The dataset is assumed to contain various demographic and household indicators from SECC

data = pd.read_excel("C:/Users/saira/Downloads/pyhtonproject.xlsx")
print(data)

# Check for null values in the dataset
print(data.isnull().sum())
print("There is no null values in dataset")

# Display descriptive statistics to understand distribution
print("Describing the dataset")
print(data[['TotalPopulation']].describe())

# Display concise summary of dataframe
print("Info of the dataset")
print(data.info())

# Display top and bottom rows of the dataset for initial glance
print(data.head)
print(data.tail)


# ------------------- Objective 1 -------------------
# Find the top 5 states with the maximum population
# Group the dataset by 'srcStateName', sum 'TotalPopulation', and sort

total_population = data.groupby("srcStateName")["TotalPopulation"].sum().sort_values()
print(total_population)

# Plotting a pie chart of the top 5 most populated states
plt.figure(figsize=(15,10))
plt.pie(total_population.tail().values, labels=total_population.tail().index, autopct='%1.1f%%')
plt.title("Top 5 States with Maximum Population")
plt.legend(total_population.tail().index, title="States", loc="center left", bbox_to_anchor=(1.1, 0.5))
plt.show()

# -------------------- Objective 2 --------------------
# Compare the number of male and female-headed households in various tehsils

tehsil_gender_data = data.groupby('Tehsilcode')[["MaleHouseholds", "FemaleHouseholds"]].sum()
print(tehsil_gender_data)

# Barplot showing male and female-headed households by Tehsil
plt.figure(figsize=(50,30))
sns.barplot(data=tehsil_gender_data, x='Tehsilcode', y='MaleHouseholds', color='blue')
sns.barplot(data=tehsil_gender_data, x='Tehsilcode', y='FemaleHouseholds', color='pink', alpha=0.6)

plt.xlabel("Tehsils")
plt.ylabel("Male and Female Households")
plt.title("Number of Male and Female-Headed Households in Various Tehsils")
plt.show()


# ------------------- Objective 3 -------------------
# Visualize the distribution of divorced persons across regions

plt.figure(figsize=(12, 6))
sns.histplot(data["DivorcedPersons"], bins=15, kde=True, color="purple")
plt.xlabel("Number of Divorced Persons")
plt.ylabel("Count of Regions")
plt.title("Distribution of Divorced Persons Across Regions")
plt.show()

# ------------------- Objective 4 -------------------
# Correlation heatmap between types of households

correlation_data = data[["TotalHouseholds", "MaleHouseholds", "FemaleHouseholds", "TransgenderHouseholds"]]
corr = correlation_data.corr()
plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Households Status Correlation Heatmap")
plt.show()

# ------------------- Objective 5 -------------------
# Analyze correlation between marital status categories

marital_status_data = data[["NeverMarriedPersons", "CurrentlyMarriedPersons", "WidowedPersons", "SeparatedPersons", "DivorcedPersons"]]
marital_corr = marital_status_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(marital_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Marital Status Correlation Heatmap")
plt.show()


# Normalization

scaler = MinMaxScaler()

data[['TotalHouseholds','TotalPopulation','MaleHouseholds']] = scaler.fit_transform(data[['TotalHouseholds','TotalPopulation','MaleHouseholds']]) 

data[['FemaleHouseholds','WomenHeadedHouseholds']]=scaler.fit_transform(data[['FemaleHouseholds','WomenHeadedHouseholds']])

data[['TransgenderHouseholds','NeverMarriedPersons','CurrentlyMarriedPersons','WidowedPersons']]=scaler.fit_transform(data[['TransgenderHouseholds','NeverMarriedPersons','CurrentlyMarriedPersons','WidowedPersons']])

data[['SeparatedPersons','DivorcedPersons','MaleHeadedHouseholds',]]=scaler.fit_transform(data[['SeparatedPersons','DivorcedPersons','MaleHeadedHouseholds',]])


print("After Normalisation\n")
print(data[['TotalHouseholds','TotalPopulation','FemaleHouseholds','TransgenderHouseholds']])
print(data[['NeverMarriedPersons','CurrentlyMarriedPersons','WidowedPersons']])
print(data[['SeparatedPersons','DivorcedPersons','MaleHeadedHouseholds','WomenHeadedHouseholds']])



# ------------------- Objective 6 -------------------
# Scatter plot showing relationship between population and households

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="TotalPopulation", y="TotalHouseholds", color='green', s=50)
plt.title("Total Population vs Total Households")
plt.xlabel("Total Population")
plt.ylabel("Total Households")
plt.grid(True)
plt.tight_layout()
plt.show()

x=data[['TotalPopulation']]
y=data[['TotalHouseholds']]
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=34)


model=LinearRegression()
model.fit(x_train,y_train)

# Predict from Model
checklungs = pd.DataFrame({'TotalPopulation':[250000000]})
result=model.predict(checklungs)
print("Predicted Total Housholds for Total Population 250M :",result)

# Plot Regression line
plt.scatter(x,y,color='blue')
plt.plot(x,model.predict(x),color='orange',linewidth=3)
plt.xlabel('Total Population')
plt.ylabel('Total Households')
plt.grid(True)
plt.title('Total Population vs Total Households')

plt.show()

# Mean Squared Error
pred=model.predict(x_test)
mse=mean_squared_error(y_test,pred)

# Shows error rate 
print(f"\nMean Squared Error (MSE): {mse:.4f}")

# R-Square Score (Goodness of fit)
r2 = r2_score(y_test,pred)
print(f"R-Square Score: {r2:.4f}")