import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

import statsmodels.api as sm
import statsmodels.formula.api as smf


class DataModelling:
    """
    Reads and cleans data in csv form using generalized rules:
    1. If column is categorical and has less than 20% missing data, drop rows
    2. If column is categorical and has more than 20% missing data, drop column
    3. If column is numerical and has less than 10% missing data, fill NAs using mean
    4. If column is numerical and has more than 10% missing data and 20%, fill NAs using mean!! (changed)
    5. If column is numerical and has more than 20% missing data, drop column
    6. If column is binary and has less than 20% missing data, drop rows
    7. If column is binary and has more than 20% missing data, drop column
    8. If column is datetime, then drop rows (no choice)
    """

    def __init__(self, path, target='Fare'):
        self.path = path
        self.df = None
        self.num_vars = []
        self.cat_vars = []
        self.target = target

    def run(self):
        """
        One-size fits all function to do all data analytics
        """
        self.read_data()  # done
        self.describe_data()  # done
        self.clean_data()  # done
        self.visualize_data()  # done
        self.linear_model()  # done

    def read_data(self):
        try:
            self.df = pd.read_csv(self.path)
        except:
            print("Please ensure that the file exists in the same directory as your Jupyter notebook :)")

    def describe_data(self):
        columns = [col for col in list(self.df.columns)]
        columns_str = ''
        for col in columns:
            columns_str += (col)
            if col == columns[-1]:
                break
            columns_str += ", "

        length = len(self.df)

        print("1. The columns in this dataframe are:")
        print(columns_str)
        print('')
        print("2. There are {} rows in this dataframe".format(length))
        print('')
        print("3. The number of missing data in this dataframe are:")
        print(self.df.isna().sum())

    def clean_data(self):
        """
        Basically dealing with NAs
        """
        cols_todrop = []
        for column in self.df.columns:
            var_type = type(self.df[column].iloc[0])
            null_vals = self.df[column].isna().sum()
            if (null_vals / len(self.df)) > 0.2:
                cols_todrop.append(column)
            elif null_vals:
                if var_type == np.float64 or var_type == np.int64:
                    self.df[column].fillna(np.mean(self.df[column]),
                                           inplace=True)  # fill mean if numerical & <10% missing

        self.df.drop(columns=cols_todrop, inplace=True)  # drop cols for cols with >10% missing
        self.df.dropna(inplace=True)  # drop rows for the rest

        # Optional: For the titanic dataset the fare classes are encoded numerically when it should have been categorical
        # Not sure how to generalize this but for next section
        self.df['Pclass'] = self.df['Pclass'].apply(str)

    def visualize_data(self):
        """
        Note: Only barplot, heatmap (correlation plot) + pairplot as this is easier to generalize
        To add: Line charts for timeseries
        """

        # define all chart types
        def plot_barcharts(col):
            """
            For categorical vars
            """
            self.df[col].value_counts().plot(kind='bar')
            plt.title("{} by frequency".format(col))
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.xticks(rotation=0)
            plt.show()

        def plot_pairplot():
            """
            For all numerical vars
            """
            sns.pairplot(self.df, vars=self.num_vars)
            plt.show()

        def plot_heatmap():
            """
            For all numerical vars
            """
            fig = plt.figure(figsize=(9, 6))

            corr = self.df[self.num_vars].corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Correlation'})
            plt.tight_layout()
            plt.yticks(rotation=0);
            plt.xticks(rotation=60)
            plt.show()

        # handle numerical and categorical vars
        for column in self.df.columns:
            var_type = type(self.df[column].iloc[0])
            if var_type == np.float64 or var_type == np.int64:
                self.num_vars.append(column)
            elif var_type == str:
                self.cat_vars.append(column)

        # plot stuff
        for col in self.cat_vars:
            if len(self.df[col].value_counts()) <= 20:  # plot only if less than 20 unique values
                plot_barcharts(col)

        plot_pairplot()
        plot_heatmap()

    def linear_model(self):
        """
        Only basic model: no interaction terms
        You have to define the target variable yourself
        We assume that we use all columns in the dataset (will give option to drop index later)
        """
        cols_likely_overfit = []
        for col in self.df.columns:
            if len(self.df[col].value_counts()) >= 20:
                cols_likely_overfit.append(col)

        if self.target in self.df.columns:
            cols_likely_overfit.append(self.target)
        else:
            print("You need to specify a target in your data columns")

        predictors = [col for col in self.df.columns if col not in cols_likely_overfit]

        formula_str = ''
        formula_str += self.target
        formula_str += ' ~ '
        for idx, col in enumerate(predictors):
            formula_str += col
            if idx == len(predictors) - 1:
                break
            formula_str += ' + '

        model = smf.ols(formula_str, data=self.df).fit()
        print(model.summary())