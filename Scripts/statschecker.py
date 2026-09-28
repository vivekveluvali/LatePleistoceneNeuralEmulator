import tensorflow as tf


class PhysicalAttractorCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,filepath="best_state_space_model.weights.h5",acf_threshold=0.97,patience=15):
        """
        initializer method for C_2 constraint callback
        ------------------------------------------------

        filepath               - file path to save the best model weights
        acf_threshold          - threshold for lag-1 ACF to accept model
        patience               - epochs without a physically-valid improvement before stopping
        """
        super(PhysicalAttractorCheckpoint,self).__init__()
        self.filepath = filepath
        self.acf_threshold = acf_threshold
        self.best_C_2 = float('inf')
        self.best_epoch = 0
        self.max_patience = patience
        self.patience = 0

    def _check_patience(self, epoch):
        if self.patience >= self.max_patience:
            self.model.stop_training = True
            print(f"\nPatience ran out. Early stopping at epoch {epoch+1}: no physically-valid "
                  f"improvement in val_C_2 for {self.patience} epochs.")

    def on_epoch_end(self,epoch=0,logs=None):

        logs = logs or {}
        # retrieve current C_2 and lag_1 ACF values
        current_C_2 = logs.get("val_C_2")
        current_acf = logs.get("val_lag1_acf")

        if current_C_2 is None or current_acf is None:
            print("\nWarning: Physical metrics not found in logs. Check test_step return keys.")
            return

        # convert from string to float
        current_C_2 = float(current_C_2)
        current_acf = float(current_acf)

        # check if ACF criteria is meet
        if current_acf < self.acf_threshold:
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}: Smoothness ({current_acf:.4f}) below threshold. Ignoring.")
            self.patience += 1
            self._check_patience(epoch)
            return

        # check if current model returns a lower C_2 constraint then previous model, if yes overwrite the current best model
        if current_C_2 < self.best_C_2:
            self.best_C_2 = current_C_2
            self.best_epoch = epoch
            self.model.save_weights(self.filepath)
            self.patience = 0
            print(f"\nEpoch {epoch+1}: NEW PHYSICAL BEST!")
            print(f" > C_2: {current_C_2:.4f}")
            print(f" > Lag-1 ACF:    {current_acf:.4f}")
            print(f" > Weights saved to {self.filepath}")
        else:
            self.patience += 1
            self._check_patience(epoch)


class WeightStatsLogger(tf.keras.callbacks.Callback):
    """Prints min/max/std of the harmonized-MSE weight tensors (1 + gamma/(fbar+eps))
    at the end of every epoch, for both the training- and validation-length trackers.
    Pools the weights across every DWT coefficient group (all decomposition levels)."""

    def _weight_stats(self, fbar_list):
        gamma, eps = self.model.dwt_gamma, self.model.dwt_eps
        weights = tf.concat([tf.reshape(1.0 + gamma / (fbar + eps), [-1]) for fbar in fbar_list], axis=0)
        return tf.reduce_min(weights), tf.reduce_max(weights), tf.math.reduce_std(weights)

    def on_epoch_end(self, epoch, logs=None):
        train_min, train_max, train_std = self._weight_stats(self.model.fbar_train)
        vali_min, vali_max, vali_std = self._weight_stats(self.model.fbar_vali)
        print(f" > hMSE weights (train): min={train_min:.4f}, max={train_max:.4f}, std={train_std:.4f}")
        print(f" > hMSE weights (vali):  min={vali_min:.4f}, max={vali_max:.4f}, std={vali_std:.4f}")
