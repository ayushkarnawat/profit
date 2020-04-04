"""Gaussian processes regressor(s)."""

import warnings
import numpy as np


BLOSUM = np.array(
    [[3.9029, 0.6127, 0.5883, 0.5446, 0.8680, 0.7568, 0.7413, 1.0569, 0.5694, 0.6325,
      0.6019, 0.7754, 0.7232, 0.4649, 0.7541, 1.4721, 0.9844, 0.4165, 0.5426, 0.9365],
     [0.6127, 6.6656, 0.8586, 0.5732, 0.3089, 1.4058, 0.9608, 0.4500, 0.9170, 0.3548,
      0.4739, 2.0768, 0.6226, 0.3807, 0.4815, 0.7672, 0.6778, 0.3951, 0.5560, 0.4201],
     [0.5883, 0.8586, 7.0941, 1.5539, 0.3978, 1.0006, 0.9113, 0.8637, 1.2220, 0.3279,
      0.3100, 0.9398, 0.4745, 0.3543, 0.4999, 1.2315, 0.9842, 0.2778, 0.4860, 0.3690],
     [0.5446, 0.5732, 1.5539, 7.3979, 0.3015, 0.8971, 1.6878, 0.6343, 0.6786, 0.3390,
      0.2866, 0.7841, 0.3465, 0.2990, 0.5987, 0.9135, 0.6948, 0.2321, 0.3457, 0.3365],
     [0.8680, 0.3089, 0.3978, 0.3015, 19.5766, 0.3658, 0.2859, 0.4204, 0.3550, 0.6535,
      0.6423, 0.3491, 0.6114, 0.4390, 0.3796, 0.7384, 0.7406, 0.4500, 0.4342, 0.7558],
     [0.7568, 1.4058, 1.0006, 0.8971, 0.3658, 6.2444, 1.9017, 0.5386, 1.1680, 0.3829,
      0.4773, 1.5543, 0.8643, 0.3340, 0.6413, 0.9656, 0.7913, 0.5094, 0.6111, 0.4668],
     [0.7413, 0.9608, 0.9113, 1.6878, 0.2859, 1.9017, 5.4695, 0.4813, 0.9600, 0.3305,
      0.3729, 1.3083, 0.5003, 0.3307, 0.6792, 0.9504, 0.7414, 0.3743, 0.4965, 0.4289],
     [1.0569, 0.4500, 0.8637, 0.6343, 0.4204, 0.5386, 0.4813, 6.8763, 0.4930, 0.2750,
      0.2845, 0.5889, 0.3955, 0.3406, 0.4774, 0.9036, 0.5793, 0.4217, 0.3487, 0.3370],
     [0.5694, 0.9170, 1.2220, 0.6786, 0.3550, 1.1680, 0.9600, 0.4930, 13.5060, 0.3263,
      0.3807, 0.7789, 0.5841, 0.6520, 0.4729, 0.7367, 0.5575, 0.4441, 1.7979, 0.3394],
     [0.6325, 0.3548, 0.3279, 0.3390, 0.6535, 0.3829, 0.3305, 0.2750, 0.3263, 3.9979,
      1.6944, 0.3964, 1.4777, 0.9458, 0.3847, 0.4432, 0.7798, 0.4089, 0.6304, 2.4175],
     [0.6019, 0.4739, 0.3100, 0.2866, 0.6423, 0.4773, 0.3729, 0.2845, 0.3807, 1.6944,
      3.7966, 0.4283, 1.9943, 1.1546, 0.3711, 0.4289, 0.6603, 0.5680, 0.6921, 1.3142],
     [0.7754, 2.0768, 0.9398, 0.7841, 0.3491, 1.5543, 1.3083, 0.5889, 0.7789, 0.3964,
      0.4283, 4.7643, 0.6253, 0.3440, 0.7038, 0.9319, 0.7929, 0.3589, 0.5322, 0.4565],
     [0.7232, 0.6226, 0.4745, 0.3465, 0.6114, 0.8643, 0.5003, 0.3955, 0.5841, 1.4777,
      1.9943, 0.6253, 6.4815, 1.0044, 0.4239, 0.5986, 0.7938, 0.6103, 0.7084, 1.2689],
     [0.4649, 0.3807, 0.3543, 0.2990, 0.4390, 0.3340, 0.3307, 0.3406, 0.6520, 0.9458,
      1.1546, 0.3440, 1.0044, 8.1288, 0.2874, 0.4400, 0.4817, 1.3744, 2.7694, 0.7451],
     [0.7541, 0.4815, 0.4999, 0.5987, 0.3796, 0.6413, 0.6792, 0.4774, 0.4729, 0.3847,
      0.3711, 0.7038, 0.4239, 0.2874, 12.8375, 0.7555, 0.6889, 0.2818, 0.3635, 0.4431],
     [1.4721, 0.7672, 1.2315, 0.9135, 0.7384, 0.9656, 0.9504, 0.9036, 0.7367, 0.4432,
      0.4289, 0.9319, 0.5986, 0.4400, 0.7555, 3.8428, 1.6139, 0.3853, 0.5575, 0.5652],
     [0.9844, 0.6778, 0.9842, 0.6948, 0.7406, 0.7913, 0.7414, 0.5793, 0.5575, 0.7798,
      0.6603, 0.7929, 0.7938, 0.4817, 0.6889, 1.6139, 4.8321, 0.4309, 0.5732, 0.9809],
     [0.4165, 0.3951, 0.2778, 0.2321, 0.4500, 0.5094, 0.3743, 0.4217, 0.4441, 0.4089,
      0.5680, 0.3589, 0.6103, 1.3744, 0.2818, 0.3853, 0.4309, 38.1078, 2.1098, 0.3745],
     [0.5426, 0.5560, 0.4860, 0.3457, 0.4342, 0.6111, 0.4965, 0.3487, 1.7979, 0.6304,
      0.6921, 0.5322, 0.7084, 2.7694, 0.3635, 0.5575, 0.5732, 2.1098, 9.8322, 0.6580],
     [0.9365, 0.4201, 0.3690, 0.3365, 0.7558, 0.4668, 0.4289, 0.3370, 0.3394, 2.4175,
      1.3142, 0.4565, 1.2689, 0.7451, 0.4431, 0.5652, 0.9809, 0.3745, 0.6580, 3.6922]]
)


class SequenceGaussianProcessRegressor:
    """Gaussian process regression (GPR) for amino acid sequences.

    The initial kernel is the prior on the dataset. After it is fit to
    some training data, the hyperparameters of the kernel are optimized
    (based off the MLL). This optimized kernel allows us to predict the
    mean and variance of the new training points.

    NOTE: Should we construct a kernel in a similar fashion to sklearn's
    RBF/C and use sklearn's GaussianProcessRegressor to train the GP? If we
    follow this convention, should the alpha (noise in the diag), beta
    (exponent), and gamma (scaling factor) be described in the kernel.
    """

    def __init__(self, kernel, alpha=0.1, beta=0.1, gamma=1):
        self.kernel_ = kernel   # substitution matrix
        self.alpha = alpha      # noise
        self.gamma = gamma      # scaling factor
        self.beta = beta        # exponential factor

        self.X_ = None
        self.K_ = None
        self.Kinv_ = None
        self.y_ = None


    def _kernel(self, Xi, Xj):
        # Check if all values in Xi, Xj are <= len(kernel)
        assert all(i < len(self.kernel_) for i in Xi)
        assert all(i < len(self.kernel_) for i in Xj)

        # Retrieve the substituion prob/covariance between the AAs based off
        # the vocab index.
        kij = np.prod(self.kernel_[[Xi, Xj]]**self.beta)
        kii = np.prod(self.kernel_[[Xi, Xi]]**self.beta)
        kjj = np.prod(self.kernel_[[Xj, Xj]]**self.beta)
        # Normalize the kernel, following Shawe-Taylor J., Cristianini N. (2004)
        # Kernel Methods for Pattern Analysis. Cambridge University Press.
        k = kij / (np.sqrt(kii*kjj))
        return np.exp(self.gamma*k)


    def fit(self, X_train, y_train, print_every=50000):
        X_train = X_train.astype(np.int)

        N = X_train.shape[0]
        total = N * (N+1) / 2
        m = 0
        self.X_ = X_train
        self.y_ = y_train

        # Compute symmetric positive-definite kernel K
        self.K_ = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                kij = self._kernel(X_train[i], X_train[j])
                # Add noise (alpha) to the diagonal elements only
                if i == j:
                    kij += self.alpha
                self.K_[i, j] = kij
                self.K_[j, i] = kij

                # Display progress
                m += 1
                if print_every is not None:
                    if m % print_every == 0:
                        print(f"Number of K elements filled: {m}/{total} "
                              f"({100. * (m/total):.2f}%)")

        # Invert kernel K; needed for prediction
        self.Kinv_ = np.linalg.inv(self.K_)
        print("Finished inverting K.")


    def perdict(self, Xstar, return_std=False, print_every=10000):
        Xstar = Xstar.astype(np.int)

        M = Xstar.shape[0]
        N = len(self.K_)
        m = 0
        total = M * N

        Kstar = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                kij = self._kernel(Xstar[i], self.X_[j])
                Kstar[i, j] = kij
                m += 1
                if print_every is not None:
                    if m % print_every == 0:
                        print(f"Number of Kstar elements filled: {m}/{total} "
                              f"({100. * (m/total):.2f}%)")
        y_mean = np.matmul(Kstar, np.matmul(self.Kinv_, self.y_))
        if return_std:
            # NOTE: The diagonal elements are all 2.718281828459045
            y_var = np.repeat(2.718, M)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(Kstar, self.Kinv_), Kstar)
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        return y_mean


    # def save(self, filepath: str) -> None:
    #     raise NotImplementedError


# def load(filepath: str) -> SequenceGaussianProcessRegressor:
#     # Should we assert that the file loaded has the same properties as original trained GP?
#     raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from profit.dataset.splitters import split_method_dict
    from profit.utils.data_utils.serializers import LMDBSerializer
    from profit.utils.data_utils.tokenizers import AminoAcidTokenizer

    dataset = LMDBSerializer.load(
        "data/3gb1/processed/lstm_fitness/primary_encoding=aa20.mdb", True)

    # Shuffle, split, and batch
    splits = ["train", "valid"]
    subset_idx = split_method_dict["stratified"]().train_valid_split(dataset[0], \
        dataset[-1].flatten(), frac_train=0.8, frac_val=0.2, n_bins=10, return_idxs=True)
    stratified = {split: [arr[idx] for arr in dataset]
                  for split, idx in zip(splits, subset_idx)}
    train_X, train_y = stratified["train"]
    val_X, val_y = stratified["valid"]

    gp = SequenceGaussianProcessRegressor(kernel=BLOSUM)
    gp.fit(train_X, train_y, print_every=50000)
    # Make prediction (mu) on whole sample space (ask for std as well)
    y_pred, sigma = gp.perdict(dataset[0], return_std=True, print_every=50000)

    tokenizer = AminoAcidTokenizer("aa20")
    seqs_4char = []
    for encoded_seq in dataset[0]:
        seq = tokenizer.decode(encoded_seq)
        seqs_4char.append(seq[38] + seq[39] + seq[40] + seq[53])
    df = pd.DataFrame(columns=["seq", "true", "pred", "sigma"])
    df["seq"] = seqs_4char
    df["true"] = dataset[-1]
    df["pred"] = y_pred
    df["sigma"] = sigma
    df["is_train"] = [1 if idx in subset_idx[0] else 0 for idx in range(dataset[0].shape[0])]

    # If x-axis is seq, sort df by seq (in alphabetical order) for "better" visualization
    # If plotting via index, no need for resorting.
    use_seq = False
    if use_seq:
        df = df.sort_values("seq", ascending=True)
    train_only = df.loc[df["is_train"] == 1]
    val_only = df.loc[df["is_train"] == 0]

    # Determine how well the regressor fit to the dataset
    rmse = np.sqrt(np.mean(np.square((val_only["pred"] - val_only["true"])))) * 1.0
    print(f"RMSE: {rmse}")

    # Plot observations, prediction and 95% confidence interval (2\sigma).
    # NOTE: We plot the whole sequence to avoid erratic line jumps
    plt.figure()
    if use_seq:
        # If using 4char amino acid seq as the x-axis values
        plt.plot(df["seq"].values, df["pred"].values, "b-", label="Prediction")
        plt.plot(df["seq"].values, df["true"].values, "r:", label="True")
        plt.plot(train_only["seq"].values, train_only["true"].values, "r.",
                 markersize=10, label="Observations")
        plt.fill(np.concatenate([df["seq"].values, df["seq"].values[::-1]]),
                 np.concatenate([df["pred"].values - 1.9600 * df["sigma"].values,
                                 (df["pred"].values + 1.9600 * df["sigma"].values)[::-1]]),
                 alpha=.5, fc="b", ec="None", label="$95\%$ confidence interval")
        plt.xticks(rotation=90)
    else:
        # If using index as the x-axis
        plt.plot(df.index, df["pred"].values, "b-", label="Prediction")
        plt.plot(df.index, df["true"].values, "r:", label="True")
        plt.plot(train_only.index, train_only["true"].values, "r.",
                 markersize=10, label="Observations")
        plt.fill(np.concatenate([df.index, df.index[::-1]]),
                 np.concatenate([df["pred"].values - 1.9600 * df["sigma"].values,
                                 (df["pred"].values + 1.9600 * df["sigma"].values)[::-1]]),
                 alpha=.5, fc="b", ec="None", label="$95\%$ confidence interval")
    plt.xlabel("Sequence ($x$)")
    plt.ylabel("Fitness ($y$)")
    plt.title("Predicting protein fitness using GPR (PDB: 3GB1)")
    plt.legend(loc="upper left")
    plt.show()
