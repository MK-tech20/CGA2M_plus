import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import copy
import itertools


class Constraint_GA2M:
    """
    Our proposed model CGA2M+.

    Attributes
    ----------
    X_train : numpy.ndarray
    y_train : numpy.ndarray
    y_train_mean : numpy.ndarray
    X_test : numpy.ndarray
    y_test : numpy.ndarray

    train_weight : np.ndarray or list
        this is same as weight in lgb.Dataset. For detail, LightGBM's documentation.

    train_weight : np.ndarray or list
        this is same as weight in lgb.Dataset. For detail, LightGBM's documentation.

    lgbm_params: dict
        parameters of LightGBM

    all_interaction_features : tuple in list
        this represents list of pairs of features, whose interaction the model may learn.  e.g. [(0,1),(0,2),(1,3)]

    monotone_constraints : list
        Monotonically increasing if `1`, monotonically decreasing if `-1`, unconstrained(No setting) if `0`.
        Assign 1, -1, or 0 to all indices for each feature in training data.

    higher_model : lightgbm.basic.Booster
        this is higer-order term.

    use_main_features : list
        indexes of features which we use when training the model. e.g.) [0,1,2,3]

    use_interaction_features : tuple in list
        indexes of pairs of features which we used when training the model. e.g.) [(0,1),(0,2),(3,4)]

    remaining_interaction_features : tuple in list
        indexes of pairs of features which we 'may' use when training the model. e.g.) [(0,1),(0,2),(3,4)]

    #We'll skip the rest.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        lgbm_params,
        monotone_constraints=None,
        monotone_constraints_method="advanced",
        all_interaction_features=None,
        train_weight=None,
        test_weight=None,
    ):
        """
        Parameters
        ----------
        X_train : numpy.ndarray

        y_train : numpy.ndarray

        y_train_mean : numpy.ndarray

        X_test : numpy.ndarray

        y_test : numpy.ndarray

        lgbm_params : dict
            parameters of LightGBM

        monotone_constraints : list
            Monotonically increasing if `1`, monotonically decreasing if `-1`, unconstrained(No setting) if `0`.
            Assign 1, -1, or 0 to all indices for each feature in training data.
            e.g.) monotone_constraints=[1,0,-1]

        monotone_constraints_method : str
            you can choose one of the following from "basic", "intermediate" and "advanced".
            This represent monotone constraints methods. See https://lightgbm.readthedocs.io/en/latest/Parameters.html.

        all_interaction_features : tuple in list
            this represents list of pairs of features, whose interaction the model may learn.  e.g. [(0,1),(0,2),(1,3)]

        train_weight : np.ndarray or list
            this is same as weight in lgb.Dataset. For detail, LightGBM's documentation.

        train_weight : np.ndarray or list
            this is same as weight in lgb.Dataset. For detail, LightGBM's documentation.
        """
        self.X_train = copy.deepcopy(X_train)
        self.y_train_mean = np.mean(y_train)
        self.y_train = copy.deepcopy(y_train) - self.y_train_mean
        self.X_test = copy.deepcopy(X_test)
        self.y_test = copy.deepcopy(y_test) - self.y_train_mean
        self.train_weight = train_weight
        self.test_weight = test_weight
        self.lgbm_params = lgbm_params
        if monotone_constraints is None:
            self.monotone_constraints = [0] * self.X_train.shape[1]
        else:
            self.monotone_constraints = monotone_constraints
            self.lgbm_params[
                "monotone_constraints_method"
            ] = monotone_constraints_method

        self.higher_model = None
        self.use_main_features = list(range(X_test.shape[1]))
        self.use_interaction_features = []

        if all_interaction_features is None:
            self.remaining_interaction_features = list(
                itertools.combinations(range(X_test.shape[1]), 2)
            )
        else:
            self.remaining_interaction_features = all_interaction_features

        self.sum_output = np.sum(np.linalg.norm(copy.deepcopy(self.y_train), ord=1))
        self.train_main_mean = {}
        self.train_interaction_mean = {}
        for i in self.use_main_features:
            self.train_main_mean[i] = 0

        for i, j in self.remaining_interaction_features:
            self.train_interaction_mean[(i, j)] = 0

    def backfitting(
        self, num_iteration=20, use_main_features=None, use_interaction_features=None
    ):
        """
        Train the model useing beckfitting algorithm.
        (When you use this package, you will not use this function directly.)

        Parameters
        ----------
        num_iteration : int
            the number of iterations of backfitting.

        use_main_features : list
            indexes of features which we use when training the model.
            e.g.) [0,1,2,3]. which means that we use x_0,x_1,...,x_3.

        use_interaction_features : tuple in list
            indexes of pairs of features which we used when training the model.
            e.g.) [(0,1),(0,2),(3,4)]
        """

        X_train = copy.deepcopy(self.X_train)
        y_train = copy.deepcopy(self.y_train)
        X_test = copy.deepcopy(self.X_test)
        y_test = copy.deepcopy(self.y_test)

        num_features = X_train.shape[1]
        self.num_features = num_features
        if use_main_features is None:
            self.use_main_features = list(range(self.num_features))
        else:
            self.use_main_features = use_main_features

        main_model_dict = {}
        main_store_train = {}
        main_store_test = {}
        for i in self.use_main_features:
            main_store_train[i] = np.zeros(X_train.shape[0])
            main_store_test[i] = np.zeros(X_test.shape[0])

        interaction_model_dict = {}
        interaction_store_train = {}
        interaction_store_test = {}
        for i, j in use_interaction_features:
            interaction_store_train[(i, j)] = np.zeros(X_train.shape[0])
            interaction_store_test[(i, j)] = np.zeros(X_test.shape[0])

        for _ in range(num_iteration):
            for i in self.use_main_features:
                main_store_train = copy.deepcopy(main_store_train)
                main_store_test = copy.deepcopy(main_store_test)

                tmp = [k for k in self.use_main_features if k != i]

                train_sum = np.zeros(X_train.shape[0])
                test_sum = np.zeros(X_test.shape[0])
                for ll in tmp:
                    train_sum = train_sum + main_store_train[ll]
                    test_sum = test_sum + main_store_test[ll]

                for k, l in use_interaction_features:
                    train_sum = train_sum + interaction_store_train[(k, l)]
                    test_sum = test_sum + interaction_store_test[(k, l)]

                residual_train = copy.deepcopy(y_train) - train_sum
                residual_test = copy.deepcopy(y_test) - test_sum

                lgb_train = lgb.Dataset(
                    copy.deepcopy(X_train[:, i]).reshape(-1, 1),
                    residual_train,
                    weight=self.train_weight,
                )
                lgb_eval = lgb.Dataset(
                    copy.deepcopy(X_test[:, i]).reshape(-1, 1),
                    residual_test,
                    reference=lgb_train,
                    weight=self.test_weight,
                )

                tmp_params = copy.deepcopy(self.lgbm_params)
                tmp_params["monotone_constraints"] = self.monotone_constraints[i]

                model = lgb.train(
                    tmp_params, lgb_train, valid_sets=lgb_eval, verbose_eval=False
                )

                pred_train = model.predict(
                    copy.deepcopy(X_train[:, i]).reshape(-1, 1),
                    num_iteration=model.best_iteration,
                )
                pred_test = model.predict(
                    copy.deepcopy(X_test[:, i]).reshape(-1, 1),
                    num_iteration=model.best_iteration,
                )
                train_main_mean = np.mean(pred_train)
                main_store_train[i] = pred_train - train_main_mean
                main_store_test[i] = pred_test - train_main_mean
                main_model_dict[i] = model
                self.train_main_mean[i] = train_main_mean

            for i, j in use_interaction_features:
                interaction_store_train = copy.deepcopy(interaction_store_train)
                interaction_store_test = copy.deepcopy(interaction_store_test)

                tmp = [(k, l) for k, l in use_interaction_features if (k, l) != (i, j)]

                train_sum = np.zeros(X_train.shape[0])
                test_sum = np.zeros(X_test.shape[0])
                for llm in self.use_main_features:
                    train_sum = train_sum + main_store_train[llm]
                    test_sum = test_sum + main_store_test[llm]

                for ll in tmp:
                    train_sum = train_sum + interaction_store_train[ll]
                    test_sum = test_sum + interaction_store_test[ll]

                residual_train = copy.deepcopy(self.y_train) - train_sum
                residual_test = copy.deepcopy(self.y_test) - test_sum

                lgb_train = lgb.Dataset(
                    copy.deepcopy(X_train[:, [i, j]]),
                    residual_train,
                    weight=self.train_weight,
                )
                lgb_eval = lgb.Dataset(
                    copy.deepcopy(X_test[:, [i, j]]),
                    residual_test,
                    reference=lgb_train,
                    weight=self.test_weight,
                )

                tmp_params = copy.deepcopy(self.lgbm_params)
                tmp_params["monotone_constraints"] = [
                    self.monotone_constraints[i],
                    self.monotone_constraints[j],
                ]

                model = lgb.train(
                    tmp_params, lgb_train, valid_sets=lgb_eval, verbose_eval=False
                )

                pred_train_int = model.predict(
                    copy.deepcopy(X_train[:, [i, j]]),
                    num_iteration=model.best_iteration,
                )
                pred_test_int = model.predict(
                    copy.deepcopy(X_test[:, [i, j]]), num_iteration=model.best_iteration
                )
                train_interaction_mean = np.mean(pred_train_int)
                interaction_store_train[(i, j)] = (
                    pred_train_int - train_interaction_mean
                )
                interaction_store_test[(i, j)] = pred_test_int - train_interaction_mean
                interaction_model_dict[(i, j)] = model
                self.train_interaction_mean[(i, j)] = train_interaction_mean

        self.main_model_dict = main_model_dict
        self.interaction_model_dict = interaction_model_dict

    def train(self, max_outer_iteration=20, backfitting_iteration=20, threshold=0.01):
        """

        Parameters
        ----------
        max_outer_iteration : int
            max_outer_iteration is the number of iterations in GA2M Framework(==Maximum number of interactions)

        backfitting_iteration : int
            the number of iterations of backfitting.

        threshold : float
            If the importance of the variable is below threshold, we consider that it is ineffective.

        Returns
        -------

        """
        count = 0
        for __ in range(max_outer_iteration):
            print(f"START {__+1}ST ITERATION")
            self.backfitting(
                num_iteration=backfitting_iteration,
                use_main_features=self.use_main_features,
                use_interaction_features=self.use_interaction_features,
            )
            residual_train = copy.deepcopy(self.y_train) - self.predict(
                self.X_train, higher_mode=False, subtract=True
            )
            residual_test = copy.deepcopy(self.y_test) - self.predict(
                self.X_test, higher_mode=False, subtract=True
            )

            if __ > 1:
                self.interaction_feature_importance()
                effect = self.interaction_output[-1]
                if effect < threshold:
                    count = count + 1
                if count >= 2:
                    print("End of training")
                    break

            residual_interaction = []
            for i, j in self.remaining_interaction_features:
                lgb_train = lgb.Dataset(
                    copy.deepcopy(self.X_train[:, [i, j]]),
                    residual_train,
                    weight=self.train_weight,
                )

                lgb_eval = lgb.Dataset(
                    copy.deepcopy(self.X_test[:, [i, j]]),
                    residual_test,
                    reference=lgb_train,
                    weight=self.test_weight,
                )

                tmp_params = copy.deepcopy(self.lgbm_params)
                model = lgb.train(
                    tmp_params, lgb_train, valid_sets=lgb_eval, verbose_eval=False
                )

                preds_train = (
                    model.predict(
                        copy.deepcopy(self.X_train[:, [i, j]]),
                        num_iteration=model.best_iteration,
                    )
                    - self.train_interaction_mean[(i, j)]
                )
                loss_train = mean_squared_error(residual_train, preds_train)

                residual_interaction.append(loss_train)

            if len(self.remaining_interaction_features) == 0:
                break
            i, j = self.remaining_interaction_features[np.argmin(residual_interaction)]
            self.remaining_interaction_features.remove((i, j))
            self.use_interaction_features.append((i, j))
            if __ == max_outer_iteration - 1:
                self.backfitting(
                    num_iteration=backfitting_iteration,
                    use_main_features=self.use_main_features,
                    use_interaction_features=self.use_interaction_features,
                )

    def predict(self, X, higher_mode=False, subtract=False):
        """
        The model makes predictions for new data.

        Parameters
        ----------
        X : numpy.ndarray
            input data

        higher_mode : bool
            If you set higher_mode = True, the model use the higher-order terms when making its predictions.

        subtract : bool
            Implementation convenience

        Returns
        -------
        preds : numpy.ndarray
            prediction
        """
        preds = np.zeros(X.shape[0])
        for i in self.use_main_features:
            preds_main = self.main_model_dict[i].predict(
                copy.deepcopy(X[:, i]).reshape(-1, 1),
                num_iteration=self.main_model_dict[i].best_iteration,
            )
            preds = preds + preds_main - self.train_main_mean[i]

        for i, j in self.use_interaction_features:
            preds_interaction = self.interaction_model_dict[(i, j)].predict(
                copy.deepcopy(X[:, [i, j]]),
                num_iteration=self.interaction_model_dict[(i, j)].best_iteration,
            )
            preds = preds + preds_interaction - self.train_interaction_mean[(i, j)]

        if higher_mode:
            preds = preds + self.higher_model.predict(
                X, num_iteration=self.higher_model.best_iteration
            )

        if not subtract:
            preds = preds + self.y_train_mean
        return preds

    def main_feature_importance(self):
        """
        Calculate the feature importance. (Note that this does not include the effect of the interaction.
        """
        main_output = []
        for i in self.use_main_features:
            x = (
                self.main_model_dict[i].predict(
                    copy.deepcopy(self.X_train[:, i]).reshape(-1, 1),
                    num_iteration=self.main_model_dict[i].best_iteration,
                )
                - self.train_main_mean[i]
            )
            main_output.append(np.linalg.norm(x, ord=1))

        self.main_output = copy.deepcopy(np.array(main_output)) / self.sum_output

    def interaction_feature_importance(self):
        """
        Calculate the feature importance of the interaction which is made by the pairs of the features.
        """
        interaction_output = []
        for i, j in self.use_interaction_features:
            x = self.interaction_model_dict[(i, j)].predict(
                copy.deepcopy(self.X_train[:, [i, j]]),
                num_iteration=self.interaction_model_dict[(i, j)].best_iteration,
            )
            interaction_output.append(np.linalg.norm(x, ord=1))

        self.interaction_output = (
            copy.deepcopy(np.array(interaction_output)) / self.sum_output
        )

    def feature_importance(self, after_prune=True, include_higher=False):
        """
        Calculate the feature importance. (Note that this considers features and the pairs of features.)

        Parameters
        ----------
        after_prune : bool
            If True, we calculate the feature importance after pruning.

        include_higher : bool
            If True, we calculate the feature importance which contain the effect of the higher-order term.

        """
        self.main_feature_importance()
        self.interaction_feature_importance()

        main_f = dict(zip(self.use_main_features, self.main_output))
        inter_f = dict(zip(self.use_interaction_features, self.interaction_output))
        main_f.update(inter_f)
        tmp_dict = copy.deepcopy(main_f)

        if include_higher:
            x = self.higher_model.predict(
                copy.deepcopy(self.X_train),
                num_iteration=self.higher_model.best_iteration,
            )
            higher_effect = np.linalg.norm(x, ord=1) / self.sum_output
            tmp_dict["higher"] = higher_effect

        if after_prune:
            self.after_feature_importance_ = tmp_dict
        else:
            self.before_feature_importance_ = tmp_dict

    def prune_and_retrain(
        self,
        threshold=0.01,
        backfitting_iteration=20,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        lgbm_params=None,
    ):
        """
        We remove the features and pairs of features whose importance is less than threshold from the model.
        We then retrain the model using only remaining features and pairs of features.

        Parameters
        ----------
        threshold : float
            If the importance of the variable is below threshold, we consider that it is ineffective.

        backfitting_iteration : int
            the number of iterations of backfitting.

        X_train : numpy.ndarray
        y_train : numpy.ndarray
        X_test : numpy.ndarray
        y_test : numpy.ndarray
        lgbm_params : dict
            parameters of LightGBM

        """
        self.feature_importance(after_prune=False, include_higher=False)

        meaningful_features = {}
        for key, val in self.before_feature_importance_.items():
            if val >= threshold:
                meaningful_features[key] = val

        self.meaningful_features = meaningful_features
        self.use_main_features = []
        self.use_interaction_features = []

        for key in self.meaningful_features.keys():
            if type(key) == type(int(1)):
                self.use_main_features.append(key)
            else:
                self.use_interaction_features.append(key)

        self.retrain(
            backfitting_iteration=backfitting_iteration,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            lgbm_params=lgbm_params,
        )

    def retrain(
        self,
        backfitting_iteration=20,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        lgbm_params=None,
    ):
        """
        We then retrain the model using only given features and pairs of features.

        Parameters
        ----------
        backfitting_iteration : int
            the number of iterations of backfitting.

        X_train : numpy.ndarray
        y_train : numpy.ndarray
        X_test : numpy.ndarray
        y_test : numpy.ndarray
        lgbm_params : dict
            parameters of LightGBM

        """
        if X_train:
            self.X_train = copy.deepcopy(X_train)
        if y_train:
            self.y_train = copy.deepcopy(y_train) - np.mean(y_train)
        if X_test:
            self.X_test = copy.deepcopy(X_test)
        if y_test:
            self.y_test = copy.deepcopy(y_test) - np.mean(y_train)
        if lgbm_params:
            self.lgbm_params = lgbm_params

        self.backfitting(
            num_iteration=backfitting_iteration,
            use_main_features=self.use_main_features,
            use_interaction_features=self.use_interaction_features,
        )

        self.feature_importance(after_prune=True)

    def higher_order_train(self):
        """
        Train the higher-order term.

        """
        residual_train = self.y_train - self.predict(
            self.X_train, higher_mode=False, subtract=True
        )
        residual_test = self.y_test - self.predict(
            self.X_test, higher_mode=False, subtract=True
        )

        lgb_train = lgb.Dataset(
            copy.deepcopy(self.X_train), residual_train, weight=self.train_weight
        )
        lgb_eval = lgb.Dataset(
            copy.deepcopy(self.X_test),
            residual_test,
            reference=lgb_train,
            weight=self.test_weight,
        )

        tmp_params = copy.deepcopy(self.lgbm_params)
        model = lgb.train(
            tmp_params, lgb_train, valid_sets=lgb_eval, verbose_eval=False
        )

        self.higher_model = model
        self.feature_importance(after_prune=True, include_higher=True)
