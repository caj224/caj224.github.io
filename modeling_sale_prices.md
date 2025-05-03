 ---
 layout: wide_default
 ---  


```python


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load data and split off X and y
housing = pd.read_csv('input_data2/housing_train.csv')
holdout = pd.read_csv('input_data2/housing_holdout.csv')


# Add feature engineering to training set
print(housing.head(2))
housing['Age'] = housing['v_Yr_Sold'] - housing['v_Year_Built']
housing['Remodeled'] = (housing['v_Year_Remod/Add'] != housing['v_Year_Built']).astype(int)
housing['TotalBathrooms'] = (housing['v_Full_Bath'] + 
                              0.5 * housing['v_Half_Bath'] + 
                              housing['v_Bsmt_Full_Bath'] + 
                              0.5 * housing['v_Bsmt_Half_Bath'])
housing['TotalSF'] = (housing['v_Total_Bsmt_SF'] + 
                      housing['v_1st_Flr_SF'] + 
                      housing['v_2nd_Flr_SF'])

# Apply same transformations to holdout set
holdout['Age'] = holdout['v_Yr_Sold'] - holdout['v_Year_Built']
holdout['Remodeled'] = (holdout['v_Year_Remod/Add'] != holdout['v_Year_Built']).astype(int)
holdout['TotalBathrooms'] = (holdout['v_Full_Bath'] + 
                              0.5 * holdout['v_Half_Bath'] + 
                              holdout['v_Bsmt_Full_Bath'] + 
                              0.5 * holdout['v_Bsmt_Half_Bath'])
holdout['TotalSF'] = (holdout['v_Total_Bsmt_SF'] + 
                      holdout['v_1st_Flr_SF'] + 
                      holdout['v_2nd_Flr_SF'])
y = np.log(housing.v_SalePrice)
housing = housing.drop('v_SalePrice',axis=1)

rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng)
```

               parcel  v_MS_SubClass v_MS_Zoning  v_Lot_Frontage  v_Lot_Area  \
    0  1056_528110080             20          RL           107.0       13891   
    1  1055_528108150             20          RL            98.0       12704   
    
      v_Street v_Alley v_Lot_Shape v_Land_Contour v_Utilities  ... v_Pool_Area  \
    0     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    
      v_Pool_QC v_Fence v_Misc_Feature v_Misc_Val v_Mo_Sold v_Yr_Sold  \
    0       NaN     NaN            NaN          0         1      2008   
    1       NaN     NaN            NaN          0         1      2008   
    
       v_Sale_Type  v_Sale_Condition  v_SalePrice  
    0          New           Partial       372402  
    1          New           Partial       317500  
    
    [2 rows x 81 columns]



```python
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

numeric_features = housing.select_dtypes(include='number').columns.tolist()

numer_pipe = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)
 
cat_features = housing.select_dtypes(include='object').columns.tolist()
cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),  
    OneHotEncoder(handle_unknown='ignore')   
)


preproc_pipe = ColumnTransformer(
    [
        ("num_impute", numer_pipe, numeric_features),
        ("cat_trans", cat_pipe, cat_features)
    ],
    remainder='drop'  
)
```


```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

ridge = Pipeline([
    ('preprocessing', preproc_pipe),
    
    ('model', Ridge()),
    

])
alphas = np.linspace(11, 11.1, 25)
strats = ['mean', 'median']

parameters = {
    'model__alpha': alphas,
    'preprocessing__num_impute__simpleimputer__strategy': strats
}
grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=parameters,
    cv=KFold(10),
    scoring='r2'
)

results = grid_search.fit(X_train, y_train)
results_df = pd.DataFrame(results.cv_results_).set_index('params')
results_df['alpha'] = [c['model__alpha'] for c in results_df.index]
results_df = results_df.sort_values('alpha')

results_df.plot(x='alpha', y='mean_test_score', kind='line',
                title='CV R^2 Score by Alpha (Ridge)')
best_ridge = results.best_estimator_

print(f"Best alpha: {results.best_params_['model__alpha']:.5f}")
print(f"Mean CV R^2 for best Ridge model: {results.best_score_:.5f}")

# Fit and test
best_ridge.fit(X_train, y_train)
y_pred = best_ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2 on test set: {r2:.5f}")

log_preds = best_ridge.predict(holdout)


predicted_prices = np.exp(log_preds)

pred_df = pd.DataFrame({
    'parcel': holdout["parcel"],  
    'prediction': predicted_prices
})




pred_df.to_csv('submission/MY_PREDICTIONS.csv', index=False)
```

    Best alpha: 11.06667
    Mean CV R^2 for best Ridge model: 0.86953
    R^2 on test set: 0.88916



    
![png](output_3_1.png)
    

