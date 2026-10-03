import os 
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import encode

warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI, Helvetica, Arial, sans-serif',
    'font.family': 'sans-serif',
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(csv_path: str = os.path.join(SCRIPT_DIR, "insurance.csv")) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape: {df_clean.shape}, nulls: {df_clean.isnull().sum().sum()}, duplicates: {df_clean.duplicated().sum()}")
    return df_clean

def distribution_of_charges(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(df["charges"], bins=40, kde=True, color="teal", ax=ax)
    ax.set_title("Distribution of Charges")
    ax.set_xlabel("Charges")

    fig_path = os.path.join(OUTPUT_DIR, "01_distribution_of_charges.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches ='tight')
    plt.close()

def distribution_of_age(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    sns.histplot(df["age"], bins=25, kde=True, color="steelblue")
    plt.title("Distribution of Age")
    fig_path = os.path.join(OUTPUT_DIR, "02_distribution_of_age.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches ='tight')
    plt.close()

def distribution_of_bmi(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    sns.histplot(df["bmi"], bins=25, kde=True, color="steelblue")
    plt.title("Distrubtion of BMI's")
    fig_path = os.path.join(OUTPUT_DIR, "03_distribution_of_bmi.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches ='tight')
    plt.close()

def distribution_of_children(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    sns.histplot(df["children"], bins=25, kde=True, color="steelblue")
    plt.title("Distribution of Children")
    fig_path = os.path.join(OUTPUT_DIR, "04_distribution_of_children.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches ='tight')
    plt.close()

def sex_count(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    ax = sns.countplot(x="sex", data=df, palette="viridis")
    plt.title("Count of sex")
    for p in ax.patches:
        ax.annotate(int(p.get_height()),(p.get_x()+ p.get_width()/ 2, p.get_height()), ha="center", va="bottom")
    
    fig_path = os.path.join(OUTPUT_DIR, "05_sex_count.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def smoke_count(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    ax = sns.countplot(x="smoker", data=df, palette="viridis")
    plt.title("Count of Smoker")
    for p in ax.patches : 
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    
    fig_path = os.path.join(OUTPUT_DIR, "06_smoker_count.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def region_count(df: pd.DataFrame):

    plt.figure(figsize=(6,4))
    ax = sns.countplot(x="region", data=df, palette="viridis")
    plt.title("Count of Regions")
    for p in ax.patches : 
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

    fig_path = os.path.join(OUTPUT_DIR, "07_region_count.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def age_vs_charges(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.7, palette={"yes":"crimson", "no" : "royalblue"})
    plt.title("Age vs Charges (by Smoker)")
    fig_path = os.path.join(OUTPUT_DIR, "08_age_vs_count.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def correlation_heatmap(df: pd.DataFrame, fe):
    df = encode(df, fe)
    plt.figure(figsize=(13, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    fig_path = os.path.join(OUTPUT_DIR, "09_heatmap.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def loss_curve(history: list):
    plt.figure(figsize=(8, 4))
    plt.plot(history, color="royalblue", linewidth=1.5)
    plt.title("Gradient Descent \u2014 Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "10_loss_curve.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def predicted_vs_actual(y_test, pred):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test, pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    fig_path = os.path.join(OUTPUT_DIR, "11_predicted_vs_actual.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def residual(pred, residual):
    plt.figure(figsize=(8,4))
    plt.scatter(pred, residual)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (error)")
    fig_path = os.path.join(OUTPUT_DIR, "12_residual.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():

    df = load_data()

    distribution_of_charges(df)
    distribution_of_age(df)
    distribution_of_bmi(df)
    distribution_of_children(df)
    sex_count(df)
    region_count(df)
    age_vs_charges(df)
    correlation_heatmap(df, fe=True)

if __name__ == "__main__":
    main()
