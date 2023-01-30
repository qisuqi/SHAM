import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap
from SHAM import utils
from SHAM.data import ProcessData
import time
import warnings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Pre define metrics for training for classification problems
CLASSIFICATION_METRICS = [tf.keras.metrics.TruePositives(name='tp'),
                          tf.keras.metrics.FalsePositives(name='fp'),
                          tf.keras.metrics.TrueNegatives(name='tn'),
                          tf.keras.metrics.FalseNegatives(name='fn'),
                          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          tf.keras.metrics.AUC(name='auc')]

# Pre define metrics for training for regression problems
REGRESSION_METRICS = [tf.keras.metrics.MeanSquaredError(name="mse"),
                      tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                      tf.keras.metrics.MeanAbsoluteError(name="mae"),
                      tf.keras.metrics.MeanAbsolutePercentageError(name="mape")]


class MLP:
    def __init__(self,
                 title,
                 epochs,
                 directory,
                 regression=True,
                 prediction=None,
                 history=None,
                 best_hps=None):

        self.title = title
        self.epochs = epochs
        self.regression = regression
        self.directory = directory

        self.prediction = prediction
        self.history = history
        self.best_hps = best_hps

    def model_builder(self, x_train, imbalanced=None):

        inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

        dense1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)
        drop1 = tf.keras.layers.Dropout(0.5)(dense3)
        flatten = tf.keras.layers.Flatten()(drop1)

        if self.regression:
            if imbalanced is not None:
                outputs = tf.keras.layers.Dense(1,
                                                activation='tanh',
                                                bias_initializer=imbalanced[1])(flatten)
            else:
                outputs = tf.keras.layers.Dense(1,
                                                activation='tanh')(flatten)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=REGRESSION_METRICS)
        else:
            if imbalanced is not None:
                outputs = tf.keras.layers.Dense(2,
                                                activation='softmax',
                                                bias_initializer=imbalanced[1])(flatten)
            else:
                outputs = tf.keras.layers.Dense(2, activation='softmax')(flatten)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=CLASSIFICATION_METRICS)

        return model

    def train_model(self, x_train, x_val, y_train, y_val, name, verbose, imbalanced=None):

        model = self.model_builder(x_train, imbalanced)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if self.regression:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.epochs / 10, mode='min')
        else:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=self.epochs / 10, mode='max')

        start = time.time()

        if imbalanced is not None:
            class_weight = imbalanced[0]
            self.history = model.fit(x_train,
                                     y_train,
                                     validation_data=(x_val, y_val),
                                     epochs=self.epochs,
                                     batch_size=2048,
                                     verbose=verbose,
                                     callbacks=[es],
                                     class_weight=class_weight)
        else:
            self.history = model.fit(x_train,
                                     y_train,
                                     validation_data=(x_val, y_val),
                                     epochs=self.epochs,
                                     batch_size=2048,
                                     verbose=verbose,
                                     callbacks=[es])
        end = time.time()
        print('Training Time: ', end - start)

        path = f"saved_models/{self.title}_bestmodel_{name}.h5"
        model.save(path)
        return path

    def predict(self, x_test, x_test_ids, scaler, normaliser, saved_model_path):

        model = tf.keras.models.load_model(saved_model_path)

        if self.regression:
            testX_ids = x_test_ids.drop(['ID', 'Date'], axis=1)

            prediction = model.predict(testX_ids)

            inv_test = normaliser.inverse_transform(prediction)
            inv_test = scaler.inverse_transform(inv_test)
            self.prediction = inv_test
        else:
            self.prediction = model.predict(x_test)

        return self.prediction

    def get_SHAPValues(self,
                       x_train_df_ids,
                       y_train_df_ids,
                       x_test_df_ids,
                       y_test_df_ids,
                       x_val_df_ids,
                       y_val_df_ids,
                       feature_names,
                       saved_model_path):

        model = tf.keras.models.load_model(saved_model_path)

        train_x = x_train_df_ids.drop(['ID', 'Date'], axis=1)
        train_x_array = np.array(train_x).astype('float32')

        test_x = x_test_df_ids.drop(['ID', 'Date'], axis=1)
        test_x_array = np.array(test_x).astype('float32')

        val_x = x_val_df_ids.drop(['ID', 'Date'], axis=1)
        val_x_array = np.array(val_x).astype('float32')

        warnings.filterwarnings("ignore")

        shap_explainer = shap.KernelExplainer(model=model, data=train_x_array[:50], link='identity')

        test_shap_values = shap_explainer.shap_values(test_x_array)
        train_shap_values = shap_explainer.shap_values(train_x_array)
        val_shap_values = shap_explainer.shap_values(val_x_array)


        shap.initjs()
        shap.summary_plot(shap_values=test_shap_values,
                          features=test_x_array,
                          feature_names=feature_names)
        shap.summary_plot(shap_values=test_shap_values[-1],
                          features=test_x_array,
                          feature_names=feature_names)

        test_shap_array = np.array(test_shap_values)
        train_shap_array = np.array(train_shap_values)
        val_shap_array = np.array(val_shap_values)

        if self.regression:
            test_shap_array = np.reshape(test_shap_array,
                                         (test_shap_array.shape[1],
                                          test_shap_array.shape[2]))
            train_shap_array = np.reshape(train_shap_array,
                                          (train_shap_array.shape[1],
                                           train_shap_array.shape[2]))
            val_shap_array = np.reshape(val_shap_array,
                                        (val_shap_array.shape[1],
                                         val_shap_array.shape[2]))
        else:
            test_shap_array = np.abs(np.mean(test_shap_array, axis=0))
            train_shap_array = np.abs(np.mean(train_shap_array, axis=0))
            val_shap_array = np.abs(np.mean(val_shap_array, axis=0))
        
        if self.regression:
            porcess_data = ProcessData()
            
            test_shap_df1 = pd.DataFrame(test_shap_array,
                                         columns=feature_names)
            train_shap_df1 = pd.DataFrame(train_shap_array,
                                          columns=feature_names)
            val_shap_df1 = pd.DataFrame(val_shap_array,
                                        columns=feature_names)
                                    
            test_shap_df1.insert(0, 'ID', x_test_df_ids['ID'])
            test_shap_df1.insert(1, 'Date', x_test_df_ids['Date'])
            
            train_shap_df1.insert(0, 'ID', x_test_df_ids['ID'])
            train_shap_df1.insert(1, 'Date', x_train_df_ids['Date'])
            
            val_shap_df1.insert(0, 'ID', x_test_df_ids['ID'])
            val_shap_df1.insert(1, 'Date', x_val_df_ids['Date'])
            
            test_shap_df = porcess_data.preprocess_timesteps(test_shap_df1,
                                                             static_cols_lens=0,
                                                             n_in=1,
                                                             n_out=1,
                                                             groupby=False,
                                                             dropnan=True)
                                                         
            train_shap_df = porcess_data.preprocess_timesteps(train_shap_df1,
                                                              static_cols_lens=0,
                                                              n_in=1,
                                                              n_out=1,
                                                              groupby=False,
                                                              dropnan=True)
                                                              
            val_shap_df = porcess_data.preprocess_timesteps(val_shap_df1,
                                                            static_cols_lens=0,
                                                            n_in=1,
                                                            n_out=1,
                                                            groupby=False,
                                                            dropnan=True)
                                        
        else:
            test_shap_df1 = pd.DataFrame(test_shap_array,
                                         columns=feature_names,
                                         index=x_test_df_ids.index)
            train_shap_df1 = pd.DataFrame(train_shap_array,
                                          columns=feature_names,
                                          index=x_train_df_ids)
            val_shap_df1 = pd.DataFrame(val_shap_array,
                                        columns=feature_names,
                                        index=x_val_df_ids)
                                    
            test_shap_df = pd.merge(test_shap_df1,
                                    y_test_df_ids,
                                    how='inner',
                                    on=test_shap_df1.index)
            test_shap_df = test_shap_df.drop('key_0', axis=1)
    
            train_shap_df = pd.merge(train_shap_df1,
                                     y_train_df_ids,
                                     how='inner',
                                     on=train_shap_df1.index)
            train_shap_df = train_shap_df.drop('key_0', axis=1)
    
            val_shap_df = pd.merge(val_shap_df1,
                                   y_val_df_ids,
                                   how='inner',
                                   on=val_shap_df1.index)
            val_shap_df = val_shap_df.drop('key_0', axis=1)

        shap_df_concat = pd.concat([train_shap_df, val_shap_df, test_shap_df],axis=0)

        shap_cols = list(shap_df_concat.columns)
        shap_cols.remove('Date')
        shap_cols.remove('ID')

        shap_df_concat = shap_df_concat[shap_cols]

        scaler = StandardScaler()
        normaliser = MinMaxScaler()

        scaled = scaler.fit_transform(shap_df_concat.values)
        normalised = normaliser.fit_transform(scaled)

        shap_df = pd.DataFrame(normalised, columns=shap_df_concat.columns)

        return shap_df

    def plot_loss_acc_classification(self):
        """Plot loss and accuracy curves."""

        utils.plot_acc(self.history.history['accuracy'],
                       self.history.history['val_accuracy'])

        utils.plot_loss(self.history.history['loss'],
                        self.history.history['val_loss'])

    def plot_loss_regression(self):
        """Plot loss curves"""

        utils.plot_loss(self.history.history['loss'],
                        self.history.history['val_loss'])

        utils.plot_rmse(self.history.history['rmse'],
                        self.history.history['val_rmse'])

