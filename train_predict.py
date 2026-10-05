"""Polynomial regression for BT2024136: var1 (Lasso, deg 5), var2 (Ridge, deg 10)."""
import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings('ignore')
D='dataset/'; R='BT2024136'
cfg={1:('lasso',5,dict(alpha=0.01,max_iter=20000,tol=1e-4)),
     2:('ridge',10,dict(alpha=1.0))}
for v,(kind,deg,kw) in cfg.items():
    tr=pd.read_csv(f'{D}{R}_train_var{v}.csv'); te=pd.read_csv(f'{D}{R}_test_var{v}.csv')
    X,y=tr.drop(columns='y').values,tr.y.values
    reg=Lasso(**kw) if kind=='lasso' else Ridge(**kw)
    m=make_pipeline(PolynomialFeatures(deg,include_bias=False),StandardScaler(),reg)
    for seed in (0,7):
        cv=KFold(5,shuffle=True,random_state=seed)
        mse=-cross_val_score(m,X,y,cv=cv,scoring='neg_mean_squared_error').mean()
        r2=cross_val_score(m,X,y,cv=cv,scoring='r2').mean()
        print(f'var{v} {kind} deg{deg} seed{seed}: CV MSE={mse:.4f} R2={r2:.4f}')
    m.fit(X,y)
    pd.DataFrame({'y':m.predict(te[X.shape[1] and tr.drop(columns='y').columns].values)}).to_csv(f'{R}_pred_var{v}.csv',index=False)
