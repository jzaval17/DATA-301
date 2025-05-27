import pandas as pd
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
pd_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
pd_df.head()

# To get hold of the target lables:

target_lbl = breast_cancer.target

