1.	import pandas as pd
2.	
3.	def load_data():
4.	    """
5.	    Loads the Iris dataset from an online source.
6.	
7.	    Returns:
8.	        pandas.DataFrame: The loaded dataset containing features and labels.
9.	    """
10.	    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
11.	    data = pd.read_csv(url)
12.	    return data
