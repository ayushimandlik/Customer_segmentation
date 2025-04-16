import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from matplotlib import style
style.use('ggplot')
warnings.filterwarnings('ignore')

df = pd.read_excel("Online Retail.xlsx")

df.drop_duplicates(inplace=True) #removing duplicated rows
df.dropna(inplace=True) # Removing missing values

print(df.describe().T)

num_rows = df.shape[0]
print("% of entries with Quantity less than 0: ",  df[df['Quantity'] <0].shape[0] * 100 / num_rows)
print("% of entries with UnitPrices less than 0: ",  df[df['UnitPrice'] <0].shape[0]  * 100 / num_rows)

# The entries with UnitPrice less than 0 can be removed. No Customer ID.
df = df[df['UnitPrice'] >=0].reset_index()

# To deal with the negative quantities, since only a small percentage of the values are negative, we can just remove them. Other methods include replacing those values with the median or mean.

df1 = df[df['Quantity'] >=0].reset_index(drop = True)

# Now lets do some fun stuff with the data... Just because!

vc = df1['InvoiceDate'].dt.time.value_counts()
times = vc.keys().tolist()
counts = vc.tolist()

def tfloat(time_dt):
    h, m, s = [int (x) for x in str(time_dt).split(":")]
    return h + (m/60) + (s/3600)

plt.figure()
plt.bar([tfloat(x) for x in times], counts)
plt.xlabel("Time of day")
plt.ylabel('Number of sales')
plt.title("Sales at different hours of the day")
plt.savefig("sales_at_times_of_day.png")

vc = df1['InvoiceDate'].dt.month.value_counts()
month = vc.keys().tolist()
counts = vc.tolist()
plt.figure()
plt.scatter(month, counts)
plt.xlabel("Month")
plt.ylabel('Number of sales')
plt.title("Total number of sales per month")
plt.savefig("sales_month.png")

# Feature engineering
df1['Total_price'] = df1['Quantity'] * df1['UnitPrice']

columns = ['Description', 'InvoiceDate', 'CustomerID', 'UnitPrice', 'Country']

# Highest spend
highest_n_cust = 10
for col in columns:
    cust_price = df1.groupby(col)['Total_price'].sum().reset_index()\
             .sort_values(by='Total_price', ascending=False)
    highest_cust = cust_price.head(n=highest_n_cust)
    plt.figure(figsize=(5,5))
    plt.xticks(rotation=90)
    plt.bar( [str(x) for x in highest_cust[col].tolist() ], highest_cust['Total_price']);
    plt.title(col)
    plt.ylabel('Total spending')
    plt.savefig("Top_10_spend_" + col + ".png")

# Most frequent spending items.
for col in columns:
    col_count = df1[col].value_counts().head(highest_n_cust)
    plt.figure()
    sb.barplot(x= col_count.values, y=[str(x) for x in col_count.index])
    plt.title(col)
    plt.xlabel("Number of Orders")
    plt.savefig("Top_10_freq_" + col + ".png")

X = df1[["Total_price", "Quantity", "UnitPrice", "Country", "Description"]]
encoder = LabelEncoder()
X["Country"] = encoder.fit_transform(X["Country"])
X["Description"] = encoder.fit_transform(X["Description"])

corr = X.corr()

plt.figure()
sb.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm")# fmt formats the displayed values to 2 decimal places
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("Heat_map_corr.png")

# Removing outliers

q1 = df1["Quantity"].quantile(0.30)
q3 = df1["Quantity"].quantile(0.70)
iqr = q3 - q1

upper_limit = q3 + (1.5 * iqr)
lower_limit = q1 - (1.5 * iqr)

df_qq = df1.loc[(df1["Quantity"] < upper_limit) & (df1["Quantity"] > lower_limit)]
q1 = df_qq["UnitPrice"].quantile(0.25)
q3 = df_qq["UnitPrice"].quantile(0.65)
iqr = q3 - q1

upper_limit = q3 + (1.5 * iqr)
lower_limit = q1 - (1.5 * iqr)

df_qup = df_qq.loc[(df_qq["UnitPrice"] < upper_limit) & (df_qq["UnitPrice"] > lower_limit)]

encoder = LabelEncoder()
df_qup["Country"] = encoder.fit_transform(df_qup["Country"])
df_qup["Description"] = encoder.fit_transform(df_qup["Description"])

X = df_qup[["Total_price", "Quantity", "UnitPrice", "Country", "Description"]]
corr = X.corr()

plt.figure()
sb.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm")# fmt formats the displayed values to 2 decimal places
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("Heat_map_corr_removing_outliers.png")


scaler = StandardScaler()  # Initialize a StandardScaler
X = scaler.fit_transform(X)  # Apply standardization to X

inertia = []

for i in range(2, 11):
    # Initialize K-Means with 'i' clusters and a fixed random state for reproducibility
    kmeans = KMeans(n_clusters=i, random_state=20)

    # Fit K-Means to the data
    kmeans.fit(X)

    # Append the inertia (WCSS) to the list
    inertia.append(kmeans.inertia_)
plt.title("Elbow plot")
plt.plot(range(2, 11), inertia)
plt.savefig("elbow_plot.png")

kmeans = KMeans(n_clusters=5, random_state=20)
df_qup['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(15,15))
sb.pairplot(df_qup, hue='Cluster', vars=['Quantity', 'UnitPrice'], palette='tab10')
plt.savefig('pairplot.png')

