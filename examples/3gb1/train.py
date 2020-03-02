"""Train 3gb1 protein engineering model."""

from profit import backend as P
from profit.dataset.splitters import split_method_dict

from data import load_dataset


# Preprocess + load the dataset
data = load_dataset('egcn', 'tertiary', labels='Fitness', num_data=50,
                    filetype='mdb', as_numpy=False)

if P.backend() == "pytorch":
    import torch
    from torch.utils.data import DataLoader, Subset
    from profit.models.pytorch.egcn import EmbeddedGCN
    from profit.utils.training_utils.pytorch.optimizers import AdamW
    from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
    from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint

    # Stratify the dataset into train/val sets
    # TODO: Use a stratified sampler to split the target labels equally into each
    # subset. That is, both the train and validation datasets will have the same
    # ratio of low/mid/high fitness variants as the full dataset in each batch.
    # See: https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907/2
    _dataset = torch.Tensor(len(data), 1) # hack to allow splitting to work properly
    _labels = [data[idx]['arr_3'].item() for idx in range(len(data))] # this op is (ANNOYINGLY) slow
    # Create subset indicies
    train_idx, val_idx = split_method_dict['stratified']().train_valid_split(
        _dataset, labels=_labels, frac_train=0.8, frac_val=0.2, return_idxs=True,
        task_type="auto", n_bins=10)
    train_dataset = Subset(data, train_idx)
    val_dataset = Subset(data, val_idx)
    # Compute sample weights

    # For now, we hope that shuffling the dataset will be enough to make it random
    # that some non-zero target labels are still introduced in each batch. Obviously
    # this is not ideal, but, for the sake of quickness, works for now.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # Initialize model
    num_atoms, num_feats = data[0]['arr_0'].shape
    model = EmbeddedGCN(num_atoms, num_feats, num_outputs=1, num_layers=1,
                        units_conv=8, units_dense=8)

    # Init callbacks
    stop_clbk = EarlyStopping(patience=2, verbose=1)
    save_clbk = ModelCheckpoint("results/3gb1/egcn_fitness/", verbose=1, 
                                save_weights_only=True, prefix="round0")
    # Cumbersome, but required to ensure weights get saved properly.
    # How do we ensure that the model (and its updated weights) are being used
    # everytime we are sampling the new batch?
    save_clbk.set_model(model)

    # Construct loss function and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = AdamW(model.parameters(), lr=1e-4)

    print('Training...')
    # PSEUDOCODE: Until the convergeg criteria is not met: i.e. acqusition func. didn't change
    # PSEUDOCODE: Update the prefix in the model saving such that it is round{idx}
    for epoch in range(5):
        # Training loop
        model.train()
        train_loss_epoch = 0
        for batch_idx, batch in enumerate(train_loader):
            # TODO: Map feats to gpu device
            atoms, adjms, dists, labels = batch.values()
            # Forward pass though model
            train_y_pred = model([atoms, adjms, dists])

            # Compute and print loss
            loss = criterion(train_y_pred, labels)
            train_loss_epoch += loss.item()
            print(batch_idx, loss.item()) # loss per mini batch

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, train_loss_epoch)

        # Validation loop
        model.eval()
        val_loss_epoch = 0
        for j, val_batch in enumerate(val_loader):
            # TODO: Map feats to gpu device
            val_atoms, val_adjms, val_dists, val_labels = val_batch.values()
            val_y_pred = model([val_atoms, val_adjms, val_dists])
            val_loss = criterion(val_y_pred, val_labels)
            val_loss_epoch += val_loss.item()
        print(epoch, val_loss_epoch)

        # Stop training (based off val loss) and save (top k) ckpts
        save_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
        should_stop = stop_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
        if should_stop:
            break

    # See active sampling technique: https://github.com/rmunro/pytorch_active_learning
    # Better, as it shows examples: https://github.com/ej0cl6/deep-active-learning/

    # PSEUDOCODE: Compute posterior probability across whole search space?
    # PSEUDOCODE: Determine which idx sample using BO, note we should set the new idx to True, for those that have been already sampled
    # PSEUDOCODE: Preprocess the new samples to be trained using the model...we also have to ensure that they have info on the 
    # PSEUDOCODE: Online train (continuing from the previous model's weights) with the new data. NOTE: Do we retrain on the whole data? Since we assume the predicted labels are the "true" labels from the first model, how will we know that we are even increasing model performance? Rather, should we condition on the posterior? See CbAS paper from Listgarten et. al. Additionally, how well the model learns and samples for the next iteration heavily depends on the inital training samples...
else:
    import numpy as np
    from profit.models.tensorflow.egcn import EmbeddedGCN

    # Shuffle, split and batch
    # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
    # https://docs.databricks.com/applications/deep-learning/data-prep/tfrecords-to-tensorflow.html
    train_idx, val_idx = split_method_dict['random']().train_valid_split(data[0], \
        labels=data[-1].flatten(), return_idxs=True)
    train_data = []
    val_data = []
    for arr in data:
        train_data.append(arr[train_idx])
        val_data.append(arr[val_idx])

    train_X = train_data[:-1]
    train_y = train_data[-1]
    val_X = val_data[:-1]
    val_y = val_data[-1]

    # Initialize eGCN model (really hacky), it also assumes we have the data loaded 
    # in memory, which is the wrong approach. Instead, we should peek into the 
    # shape defined in the TF tensors.

    # NOTE: Only use when TfRecordsDataset() (i.e. as_numpy=False) is used
    # num_atoms, num_feats = list(map(int, data.output_shapes[0]))
    # num_outputs = list(map(int, data.output_shapes[0]))[0]
    num_atoms, num_feats = train_data[0].shape[1], train_data[0].shape[2]
    labels = train_data[-1]
    num_outputs = labels.shape[1]
    labels_std = np.std(labels, axis=0)
    model = EmbeddedGCN(num_atoms, num_feats, num_outputs=num_outputs, std=labels_std).get_model()

    # Fit model and report metrics
    model.fit(train_X, train_y, batch_size=5, epochs=3, shuffle=True,
              validation_data=(val_X, val_y), verbose=1)