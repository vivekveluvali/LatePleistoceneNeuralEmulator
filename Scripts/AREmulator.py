import numpy as np
import tensorflow as tf

class RolloutEmulator(tf.keras.Model):
    """
    custom keras model class for rollout training, validation and prediction

    Loss: Harmonized MSE (hMSE) computed on a multi-level (default 2-level) Haar
    DWT of the rollout trajectory, per "The Procrustean Bed of Time Series"
    (arXiv:2512.18610), Eq. 21-23. The DWT is implemented with tf.nn.conv1d using
    fixed (non-trainable) Haar filter taps so it stays differentiable under
    GradientTape - PyWavelets operates outside the TF graph and would break
    gradient flow. Multi-level decomposition is the standard recursive "pyramid
    algorithm": each level decomposes the previous level's approximation into a
    new, shorter approximation plus a detail band; every level's detail band is
    kept, and only the final (coarsest) approximation is kept from the last level.
    """
    def __init__(self,brain,rollout_length,vali_rollout_length,window_size,n_features,
                 L_2_weight=1.0,dwt_gamma=0.5,dwt_beta=0.3,dwt_eps=1e-6,dwt_levels=3,
                 noise_level=1e-2,gp_noise_scale=1.0,vali_gp_noise=False):
        """
        initializer method for the rollout trainig class
        ------------------------------------------------

        brain               - model to train via rollout training
        rollout_length      - training rollout length
        vali_rollout_length - validation rollout length
        window_size         - input window length (used to build the brain)
        n_features          - number of input columns (X, Y, Z, Insol -> 4)
        L_2_weight          - weight on the harmonized MSE loss (default 1.0)
        dwt_gamma           - Eq. 21 gamma: balances the flat vs. relative-error term (paper default 0.5)
        dwt_beta            - Eq. 23 beta: EMA smoothing factor for the historical-magnitude tracker (paper default 0.3)
        dwt_eps             - Eq. 21 epsilon: minimum threshold preventing weight explosion
        dwt_levels          - number of recursive DWT decomposition levels (default 3)
        noise_level         - standard deviation of normally distributed noise used for noise injection during training.
                              NOTE: train_step is a tf.function, so this value is baked in when it is first
                              traced (first .fit call); set it here rather than changing it afterwards.
        gp_noise_scale      - multiplier on the GPR-uncertainty noise (see _gp_perturb); 1.0 draws with each record's
                              own GPR std, 0.0 turns it off. Only active when the GPR stds are passed to .fit()
                              as extra inputs. Baked in at tracing like noise_level.
        vali_gp_noise       - also perturb the validation data with GPR noise (default False: validation, and so the
                              val_C_2 / val_lag1_acf checkpoint gate, is scored against the GPR mean trajectory)
        """

        super(RolloutEmulator, self).__init__()
        self.brain = brain
        self.rollout_length = rollout_length
        self.vali_rollout_length = vali_rollout_length
        self.L_2_weight = L_2_weight
        self.noise_level = noise_level
        self.gp_noise_scale = gp_noise_scale
        self.vali_gp_noise = vali_gp_noise

        self.brain.build((None, window_size, n_features))

        # --- Harmonized MSE (hMSE) setup --------------------------------------
        # Haar wavelet: orthonormal 2-tap low-pass (approximation) / high-pass (detail)
        # filters, shaped (filter_width, in_channels=1, out_channels=1) for conv1d.
        self.dwt_low  = tf.constant([[[1/np.sqrt(2)]], [[ 1/np.sqrt(2)]]], dtype=tf.float32)
        self.dwt_high = tf.constant([[[1/np.sqrt(2)]], [[-1/np.sqrt(2)]]], dtype=tf.float32)

        self.dwt_gamma = dwt_gamma
        self.dwt_beta = dwt_beta
        self.dwt_eps = dwt_eps
        self.dwt_levels = dwt_levels

        # historical-magnitude EMA state (f_bar in Eq. 23), one tracker per DWT
        # coefficient group (each level's detail band, plus the final coarsest
        # approximation), one value per (coefficient index, state variable).
        # train_step and vali_step roll out to different lengths, so each needs
        # its own set of trackers sized to its own coefficient-group lengths.
        # Haar DWT with stride 2 / VALID padding on length L gives L//2
        # coefficients per level (an odd L would floor-truncate the last sample -
        # both rollout lengths here are even at every level this goes to, so this
        # doesn't come up).
        for L in (self.rollout_length, self.vali_rollout_length):
            if L % (2 ** self.dwt_levels) != 0:
                raise ValueError(f"rollout length {L} not divisible by 2**dwt_levels={2**self.dwt_levels}; "
                                 "the Haar DWT would silently drop samples")

        train_group_lengths = self._dwt_multilevel_lengths(self.rollout_length, self.dwt_levels)
        vali_group_lengths = self._dwt_multilevel_lengths(self.vali_rollout_length, self.dwt_levels)

        self.fbar_train = [tf.Variable(tf.ones((L, 3)), trainable=False, name=f"fbar_train_{i}")
                            for i, L in enumerate(train_group_lengths)]
        self.fbar_vali = [tf.Variable(tf.ones((L, 3)), trainable=False, name=f"fbar_vali_{i}")
                           for i, L in enumerate(vali_group_lengths)]

    # returns number of (trainable) variables/ size of the neural network
    @property
    def trainable_variables(self):
        return self.brain.trainable_variables

    # call to model to predict dependent on the inputs (training set to None)
    def call(self, inputs, training=None):
        return self.brain(inputs, training=training)

    def _dwt(self, x, length):
        """
        Shallow (1-level) Haar DWT applied independently to each of the 3 state
        channels (X, Y, Z), via tf.nn.conv1d with the channel dimension folded
        into the batch dimension so channels aren't mixed together.

        x: (batch, length, 3) -> cA, cD: (batch, length//2, 3)
        """
        batch = tf.shape(x)[0]
        coeff_len = length // 2

        x_ch = tf.transpose(x, [0, 2, 1])              # (batch, 3, length)
        x_ch = tf.reshape(x_ch, [-1, length, 1])        # (batch*3, length, 1)

        cA = tf.nn.conv1d(x_ch, self.dwt_low,  stride=2, padding="VALID")  # (batch*3, coeff_len, 1)
        cD = tf.nn.conv1d(x_ch, self.dwt_high, stride=2, padding="VALID")

        cA = tf.transpose(tf.reshape(cA, [batch, 3, coeff_len]), [0, 2, 1])  # (batch, coeff_len, 3)
        cD = tf.transpose(tf.reshape(cD, [batch, 3, coeff_len]), [0, 2, 1])
        return cA, cD

    def _dwt_multilevel(self, x, length, levels):
        """
        Recursive multi-level (Mallat pyramid) Haar DWT, built on top of the
        single-level _dwt above: at each level, the current approximation is
        decomposed into a new, shorter approximation plus a detail band; the
        detail band is kept and the approximation is fed into the next level.

        x: (batch, length, 3)
        returns: a list of (batch, len_i, 3) tensors, ordered
                 [cD_level1, cD_level2, ..., cD_levelN, cA_levelN]
                 i.e. detail coefficients from finest to coarsest scale, followed
                 by the final (coarsest) approximation.
        """
        coeffs = []
        current, current_length = x, length
        for _ in range(levels):
            cA, cD = self._dwt(current, current_length)
            coeffs.append(cD)
            current, current_length = cA, current_length // 2
        coeffs.append(current)  # final coarse approximation
        return coeffs

    @staticmethod
    def _dwt_multilevel_lengths(length, levels):
        """Coefficient-group lengths for _dwt_multilevel, in the same order it returns them."""
        lengths = []
        current_length = length
        for _ in range(levels):
            current_length = current_length // 2
            lengths.append(current_length)
        lengths.append(current_length)  # final approximation shares the last level's length
        return lengths

    def _harmonized_mse(self, true_seq, pred_seq, length, fbar_list, update):
        """
        Harmonized MSE (Eq. 21) computed on a multi-level Haar DWT of true_seq/pred_seq.

        true_seq, pred_seq - (batch, length, 3) ground-truth / predicted trajectories
        fbar_list           - list of (len_i, 3) EMA trackers, one per DWT coefficient
                               group (see _dwt_multilevel for the group ordering)
        update               - whether to update fbar_list with this batch (Eq. 23)
        """
        true_groups = self._dwt_multilevel(true_seq, length, self.dwt_levels)
        pred_groups = self._dwt_multilevel(pred_seq, length, self.dwt_levels)

        loss = 0.0
        for true_g, pred_g, fbar in zip(true_groups, pred_groups, fbar_list):
            if update:
                batch_mag = tf.reduce_mean(tf.abs(true_g), axis=0)  # (len_i, 3)
                fbar.assign(self.dwt_beta * fbar + (1 - self.dwt_beta) * batch_mag)

            weight = 1.0 + self.dwt_gamma / (fbar + self.dwt_eps)  # (len_i, 3)
            loss += tf.reduce_mean(weight * tf.square(true_g - pred_g))

        return loss

    def _gp_perturb(self, features, X_init, y):
        """
        Replaces the GPR mean trajectory with one noisy realization of it: independent Gaussian noise for each
        record (X, Y, Z) and time step, with that record's GPR std at that time step. Fresh noise is drawn for
        every batch of every epoch. Insolation is deterministic and is left untouched.

        The stds come in through .fit() as extra inputs (LPNE.create_std_windows):
            features[2] - std_init,   (batch, window_size, 3): std of each state value in the input window
            features[3] - std_future, (batch, rollout_length, 3): std of each target value
        If they aren't passed (only [X, forcing]), X_init and y are returned unchanged.

        X_init - (batch, window_size, 4) input window; y - (batch, rollout_length, 3) target trajectory
        """
        if len(features) < 4 or self.gp_noise_scale == 0:
            return X_init, y

        std_init = tf.cast(features[2], X_init.dtype) * self.gp_noise_scale
        std_future = tf.cast(features[3], y.dtype) * self.gp_noise_scale

        noisy_states = X_init[:, :, 0:3] + tf.random.normal(tf.shape(std_init), dtype=X_init.dtype) * std_init
        X_init = tf.concat([noisy_states, X_init[:, :, 3:]], axis=-1)
        y = y + tf.random.normal(tf.shape(std_future), dtype=y.dtype) * std_future
        return X_init, y

    # heart of the actual training step
    @tf.function
    def train_step(self,data):

        # as you can see here, the training step will recieve data in a tuple of (features,data), which is equivalent to the data you pass to the
        # x= ... and y= .. argument in the keras.Model.fit(...) method
        (features,y) = data

        # features itself contains the X_init values being the values of the state space (X,Y,Z) and the insolation forcing in the input window w at index 0
        # at index 1 the "future" insolation forcing aka. the insolation during the rollout is passed with shape (rollout_length,1)
        # optional indices 2 and 3 are the GPR stds of the window and the target (see _gp_perturb)
        X_init = features[0]
        future_forcing = features[1]

        # swap the GPR mean for a noisy realization of it (no-op if the stds weren't passed)
        X_init, y = self._gp_perturb(features, X_init, y)

        batch_size = tf.shape(X_init)[0]
        n_features = tf.shape(X_init)[2]

        # scale augmentation to randomly rescale state space feature and target trajectories
        scale = tf.random.uniform([batch_size,1,1], 0.8, 1.2)

        # apply scaling to (X,Y,Z) in the input window
        X_init_scaled = tf.concat([X_init[:, :, 0:3] * scale, X_init[:, :, 3:]], axis=-1)
        y_scaled = y * scale

        # tell tensorflow to create the context manager of the variables created in the with statement and tracks them to calculate the gradients
        with tf.GradientTape() as tape:

            # passing input window as first window to use in rollout training
            current_window = X_init_scaled

            # calculate ground truth trajectory
            true_path = tf.concat([X_init_scaled[:, -1:, 0:3], y_scaled], axis=1)

            all_paths = []

            # actual rollout training
            for i in range(self.rollout_length):

                # current window (first step this is just the input window) with added normally distributed (input) noise
                noisy_window = current_window + tf.random.normal(tf.shape(current_window), stddev=self.noise_level)

                # generating increment predictions (forward-pass)
                d_prediction = self.brain(noisy_window, training=True)

                # adding predicted increment to last entry in the crurrent windoe
                x_now = current_window[:, -1, 0:3]
                prediction_abs = x_now + d_prediction

                all_paths.append(prediction_abs)

                # adding additional output noise
                prediction_abs += tf.random.normal(tf.shape(prediction_abs), stddev=self.noise_level)

                next_step = tf.zeros((batch_size, 1, n_features))

                pred = tf.expand_dims(prediction_abs, axis=1)

                forc_true = future_forcing[:, i]

                # noise injection usedto prevent NN from remembering just the values, resulting in insolation just being used as a clock (safety measure)
                forc_noisy = forc_true + tf.random.normal(tf.shape(forc_true), stddev=self.noise_level)

                forc = tf.reshape(forc_noisy, (batch_size, 1, 1))
                next_step = tf.concat([pred,forc],axis=-1)

                # appending next predicted state and insolation forcig to the current window and moving window one time step forward
                current_window = tf.concat([current_window[:,1:,:],next_step],axis=1)

            # creating stack over rollout window for trajectory
            p_stack = tf.stack(all_paths, axis=1)

            # Harmonized MSE, computed on a shallow Haar DWT of the trajectory (Eq. 21)
            abs_loss = self._harmonized_mse(true_path[:,1:,:], p_stack, self.rollout_length,
                                             self.fbar_train, update=True)
            total_loss = self.L_2_weight * abs_loss

            # plain point-to-point MSE, logged for comparison only - not part of total_loss,
            # so it has no effect on the gradients computed below
            pointwise_mse = tf.reduce_mean(tf.square(true_path[:,1:,:] - p_stack))

        # Backporpagation through the forward pass operations tracked by GradientTape()
        grads = tape.gradient(total_loss,self.trainable_variables)
        # weight update
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {
            "loss": total_loss,
            "abs": abs_loss,
            "mse": pointwise_mse}

    # test_step used in the validation process
    # essentially the same as the training step, only with training disabled and no backpropagation/ weights update
    def test_step(self,data):

        def calculate_lag1_acf(p_stack):
            """
            calculate the lag_1 auto-correlation function to ensure causality in the validation step is preserved
            --------------------------------------------------

            p_stack - predicted validation rollout trajetory
            """

            mu = tf.reduce_mean(p_stack, axis=1, keepdims=True)
            var = tf.math.reduce_variance(p_stack, axis=1, keepdims=True)

            p_centered = p_stack - mu
            t0 = p_centered[:, :-1, :]
            t1 = p_centered[:, 1:, :]

            acf_per_sample = tf.reduce_mean(t0 * t1, axis=1, keepdims=True) / (var + 1e-7)

            return tf.reduce_mean(acf_per_sample)


        (features,y) = data
        X_init = features[0]
        future_forcing = features[1]

        # validation is scored against the GPR mean unless vali_gp_noise is set
        if self.vali_gp_noise:
            X_init, y = self._gp_perturb(features, X_init, y)

        batch_size = tf.shape(X_init)[0]
        n_features = tf.shape(X_init)[2]

        current_window = X_init

        all_paths = []

        true_path = tf.concat([X_init[:, -1:, 0:3], y], axis=1) # [Batch, Rollout+1, 3]

        for i in range(self.vali_rollout_length):
            d_prediction = self.brain(current_window, training=False)

            x_now = current_window[:, -1, 0:3]
            prediction_abs = x_now + d_prediction

            all_paths.append(prediction_abs)

            next_step = tf.zeros((batch_size, 1, n_features))

            pred = tf.expand_dims(prediction_abs, axis=1)
            forc = tf.reshape(future_forcing[:, i], (batch_size, 1, 1))
            next_step = tf.concat([pred,forc],axis=-1)

            current_window = tf.concat([current_window[:,1:,:],next_step],axis=1)

        p_stack = tf.stack(all_paths, axis=1)

        # Harmonized MSE on the validation-length rollout - its own EMA tracker,
        # since the coefficient count differs from the training rollout length.
        # Updated here too (each validation pass) so its weighting is actually
        # adaptive rather than frozen at the initial f_bar=1 value; the paper
        # describes the EMA as updating "at each training iteration," so switch
        # update=False here if you want strictly training-only magnitude tracking.
        abs_loss = self._harmonized_mse(true_path[:,1:,:], p_stack, self.vali_rollout_length,
                                         self.fbar_vali, update=True)

        # plain point-to-point MSE, logged for comparison only
        pointwise_mse = tf.reduce_mean(tf.square(true_path[:,1:,:] - p_stack))

        lag1_acf = calculate_lag1_acf(p_stack)

        # calculate standard deviation for ground truth and prediction in validation step
        y_std = tf.math.reduce_std(y,axis = 1)
        p_std = tf.math.reduce_std(p_stack, axis=1)

        # calculate C_2 constraint
        C_2 =  tf.reduce_mean(tf.abs(1.0 - (p_std / (y_std + 1e-6))))

        total_loss = self.L_2_weight * abs_loss

        return {
            "loss": total_loss,
            "abs": abs_loss,
            "mse": pointwise_mse,
            "lag1_acf": lag1_acf,
            "C_2": C_2}
    
    