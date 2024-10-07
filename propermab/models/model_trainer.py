# Copyright 2024 Regeneron Pharmaceuticals Inc. All rights reserved.
#
# License for Non-Commercial Use of PROPERMAB code
#
# All files in this repository (“source code”) are licensed under the following terms below:
#
# “You” refers to an academic institution or academically employed full-time personnel only.
#
# “Regeneron” refers to Regeneron Pharmaceuticals, Inc.
#
# Regeneron hereby grants You a right to use, reproduce, modify, or distribute the PROPERMAB source 
# code, in whole or in part, whether in original or modified form, for academic research purposes only. 
# The foregoing right is royalty-free, worldwide (subject to applicable laws of the United States), 
# revocable, non-exclusive, and non-transferable.
#
# Prohibited Uses: The rights granted herein do not include any right to use by commercial entities 
# or commercial use of any kind, including, without limitation, (1) any integration into other code 
# or software that is used for further commercialization, (2) any reproduction, copy, modification 
# or creation of a derivative work that is then incorporated into a commercial product or service or 
# otherwise used for any commercial purpose, (3) distribution of the source code, in whole or in part, 
# or any resulting executables, in any commercial product, or (4) use of the source code, in whole 
# or in part, or any resulting executables, in any commercial online service.
#
# Except as expressly provided for herein, nothing in this License grants to You any right, title or 
# interest in and to the intellectual property of Regeneron (either expressly or by implication or estoppel).  
# Notwithstanding anything else in this License, nothing contained herein shall limit or compromise 
# the rights of Regeneron with respect to its own intellectual property or limit its freedom to practice 
# and to develop its products and product candidates.
#
# If the source code, whole or in part and in original or modified form, is reproduced, shared or 
# distributed in any manner, it must (1) identify Regeneron Pharmaceuticals, Inc. as the original 
# creator, (2) retain any copyright or other proprietary notices of Regeneron, (3) include a copy 
# of the terms of this License.
#
# TO THE GREATEST EXTENT PERMITTED UNDER APPLICABLE LAW, THE SOURCE CODE (AND ANY DOCUMENTATION) IS 
# PROVIDED ON AN “AS-IS” BASIS, AND REGENERON PHARMACEUTICALS, INC. EXPRESSLY DISCLAIMS ALL 
# REPRESENTATIONS, WARRANTIES, AND CONDITIONS WITH RESPECT THERETO OF ANY KIND CONCERNING THE SOURCE 
# CODE, IN WHOLE OR IN PART AND IN ORIGINAL OR MODIFIED FORM, WHETHER EXPRESS, IMPLIED, STATUTORY, OR 
# OTHER REPRESENTATIONS, WARRANTIES AND CONDITIONS, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS, 
# ACCURACY, COMPLETENESS, RIGHT TO QUIET ENJOYMENT, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR 
# NOT KNOWN OR DISCOVERABLE.  REGENERON DOES NOT WARRANT THAT THE SOURCE CODE WILL OPERATE IN AN 
# UNINTERRUPTED FASHION AND DATA MAY BE LOST OR UNRECOVERABLE. IN THE EVENT ANY OF THE PRIOR DISCLAIMERS 
# ARE UNENFORCEABLE UNDER APPLICABLE LAW, THE LICENSES GRANTED HEREIN WILL IMMEDIATELY BE NULL AND 
# VOID AND YOU SHALL IMMEDIATELY RETURN TO REGENERON THE SOURCE CODE OR DESTROY IT.
#
# IN NO CASE SHALL REGENERON BE LIABLE FOR ANY LOSS, CLAIM, DAMAGE, OR EXPENSES, OF ANY KIND, WHICH 
# MAY ARISE FROM OR IN CONNECTION WITH THIS LICENSE OR THE USE OF THE SOURCE CODE. YOU WAIVE AND 
# RELEASE REGENERON FOREVER FROM ANY LIABILITY AND YOU SHALL INDEMNIFY AND HOLD REGENERON, ITS AFFILAITES 
# AND ITS AND THEIR EMPLOYEES AND AGENTS HARMLESS FROM ANY LOSS, CLAIM, DAMAGE, EXPENSES, OR LIABILITY, 
# OF ANY KIND, FROM A THIRD-PARTY WHICH MAY ARISE FROM OR IN CONNECTION WITH THIS LICENSE OR YOUR USE 
# OF THE SOURCE CODE.

# You agree that this License and its terms are governed by the laws of the State of New York, without 
# regard to choice of law rules and the United Nations Convention on the International Sale of Goods 
# shall not apply.
#
# Please reach out to Regeneron Pharmaceuticals Inc./Administrator relating to any non-academic or 
# commercial use of the source code.
import numpy as np
import pandas as pd

from scipy import stats
from sklearn import metrics
from sklearn import model_selection


class ModelTrainer:
    def __init__(
        self, 
        model, 
        param_grid: dict, 
        X, 
        y, 
        task:  str='regression',
        refit_criterion: str='pearson_r'
    ) -> None:
        """_summary_

        Parameters
        ----------
        model : sklearn estimator
            A sklearn estimator.
        param_grid : dict
            Values of hyper-parameters to be searched.
        X : NumPy array
            Feature matrix.
        y : NumPy array
            Target.
        task : str, optional
            What the ML task is, by default 'regression'
        """
        self.model = model
        self.param_grid = param_grid
        self.X = X
        self.y = y
        self.task = task
        self.final_model = None
        self.refit_criterion = refit_criterion

    @staticmethod
    def pearson_r(y_true, y_pred):
        """Compute the Pearson R between true values and predictions.
        
        Parameters
        ----------
        y_true : list or NumPy 1d array
            True values.
        y_pred : list or NumPy 1d array
            Predicted values.
            
        Returns
        -------
        float
            The Pearson R between true values and predictions.
        """
        return stats.pearsonr(y_true, y_pred)[0]

    @staticmethod
    def spearman_rho(y_true, y_pred):
        """Compute the Spearman rho between true values and predictions.
        
        Parameters
        ----------
        y_true : list or NumPy 1d array
            True values.
        y_pred : list or NumPy 1d array
            Predicted values.
            
        Returns
        -------
        float
            The Spearman rho between true values and predictions.
        """
        return stats.spearmanr(y_true, y_pred)[0]

    def train_best_regressor(
        self, X=None, y=None, gridcv_k=5, is_final=False
    ):
        """Uses GridSearchCV to search for the best hyperparameters, then refits
        the model to the whole dataset with the best hyperparameters.
        
        Parameters
        ----------
        model
            A sklearn estimator with the usual sklearn estimator API.
        param_grid : dict
            A dict specifying the hyperparameter space to be searched.
        X : np.ndarray
            Feature matrix.
        y : np.array
            Target values.
        gridcv_k : int
            Number of cv folds to try in GridSearchCV.

        Returns
        -------
        GridSearchCV
            A GridSearchCV object fit to the whole dataset with the best
            hyperparameters.
            
        """
        # make some scorers
        mae_scorer = metrics.make_scorer(
            metrics.mean_absolute_error, greater_is_better=False
        )
        pearson_r_scorer = metrics.make_scorer(
            self.pearson_r, greater_is_better=True
        )
        spearman_rho_scorer = metrics.make_scorer(
            self.spearman_rho, greater_is_better=True
        )
        
        regr_grid_cv = model_selection.GridSearchCV(
            estimator=self.model, param_grid=self.param_grid, cv=gridcv_k, 
            scoring={
                'pearson_r': pearson_r_scorer, 
                'spearman_rho': spearman_rho_scorer,
                'mae': mae_scorer
            }, refit=self.refit_criterion
        )

        # fit to training data
        if X is None or y is None:
            regr_grid_cv.fit(self.X, self.y)
        else:
            regr_grid_cv.fit(X, y)

        if is_final:
            self.final_model = regr_grid_cv.best_estimator_
        
        return regr_grid_cv


    def run_nested_kfold(
        self, 
        k: int=5, 
        gridcv_k: int=5, 
        random_state: int=42
    ):
        """Run a nested k-fold cross validation experiment. The inner loop uses
        GridSearchCV to find the best set of hyperparameters. The model is then
        refit using the entire (k-1) folds available in the inner loop. The generalization 
        ability of the refit model is estimated using the held-out fold in the 
        outer loop.
        
        Parameters
        ----------
        k : int, optional
            Number of folds, by default 5.
        gridcv_k : int, optional
            Number of folds in GridSearchCV, bu default 5.
        random_state: int, optional
            Random seed for the KFold split.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame object with one row that has the performance metrics
            estimated via this nested k-fold cross-validation approach.
            
        """
        pearson_rs = []
        spearman_rhos = []
        maes = []
        
        outer_kfold = model_selection.KFold(
            n_splits=k, shuffle=True, random_state=random_state
        )     
        for train_idx, test_idx in outer_kfold.split(self.X, self.y):
            # split the dataset
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X[test_idx]
            y_test = self.y[test_idx]
            
            # fit to training data        
            regr_grid_cv = self.train_best_regressor(
                X=X_train, y=y_train, gridcv_k=gridcv_k)

            # predict
            test_preds = regr_grid_cv.predict(X_test)
            
            pearson_rs.append(self.pearson_r(y_test, test_preds))
            spearman_rhos.append(self.spearman_rho(y_test, test_preds))
            maes.append(metrics.mean_absolute_error(y_test, test_preds))
        
        kfold_perf_metrics = {
            'pearson_r': [np.mean(pearson_rs)],
            'spearman_rho': [np.mean(spearman_rhos)],
            'mae': [np.mean(maes)]
        }
        
        return pd.DataFrame(kfold_perf_metrics)

    def run_loocv(self, gridcv_k=5):
        """Run a nested leave-one-out cross validation experiment. The inner loop uses
        GridSearchCV with `gridcv_k` fold to find the best set of hyperparameters.
        The model is then refit using the entire (N-1) data points available in the inner loop. 
        The the refit model is then used to make predictions for the 1 hold-out data point
        in the outer loop.
        
        Parameters
        ----------
        gridcv_k : int, optional
            Number of folds in GridSearchCV, bu default 5.

        Returns
        -------
        pd.DataFrame
            Each row is a pair of true value and the LOOCV prediction.
            
        """
        test_preds = []
        test_ys = []
        
        loocv = model_selection.LeaveOneOut()
        for _, (train_idx, test_idx) in enumerate(loocv.split(self.X, self.y)):
            # split the dataset
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X[test_idx]
            y_test = self.y[test_idx]
            test_ys.extend(y_test)

            # fit to training data
            regr_grid_cv = self.train_best_regressor(
                X=X_train, y=y_train, gridcv_k=gridcv_k)
            y_pred = regr_grid_cv.predict(X_test)
            test_preds.extend(y_pred)

        loocv_df = pd.DataFrame({
            'measured': test_ys,
            'predicted': test_preds
        })
                
        return loocv_df