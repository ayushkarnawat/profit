from typing import Optional

from keras.layers import Add, Concatenate, Dense, Dropout, Flatten, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from profit.layers.convs import GraphConvS, GraphConvV
from profit.layers.embed import GraphEmbed
from profit.layers.gather import GraphGather
from profit.layers.nodes import GraphSToS, GraphSToV, GraphVToS, GraphVToV
from profit.utils.training_utils.metrics import std_mae, std_rmse, std_r2


class GCN(object):
    """Graph Convolution Network.

    Params:
    -------
    num_atoms: int
        Max number of atoms in a molecule.

    num_feats: int
        Number of features per atom in the molecule.

    num_outputs: int
        Number of outputs. This is simply the number of labels.

    units_conv: int, optional, default=128
        Number of filters/units to use in the convolution layer(s).

    units_dense: int, optional, default=128
        Number of filters/units to use in the fully connected layer(s).

    num_layers: int, optional, default=2
        Total number of graph convolution layer(s).

    task: str, optional, default='regression'
        Whether the task is classification (binary or multi-label) or 
        regression-based. 

    loss: str, optional, default="mse"
        Loss function. See keras.losses for list of all available losses.

    pooling: str, optional, default="max"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. The representations 
        are either summed, averaged, or maxed depending on the pooling 
        type chosen.

    std: float or None, optional, default=None
        Standard deviation of the dataset. If regression-based task, 
        then a std must be provided, which is usually the standard 
        deviation of the training dataset.
    """

    def __init__(self, num_atoms: int, num_feats: int, num_outputs: int,
                 units_conv: Optional[int]=128, units_dense: Optional[int]=128, 
                 num_layers: Optional[int]=2, task: Optional[str]="regression",
                 loss: Optional[str]="mse", pooling: Optional[str]="max", 
                 std: Optional[float]=None):
        self.num_atoms = num_atoms
        self.num_feats = num_feats
        self.num_outputs = num_outputs
        self.units_conv = units_conv
        self.units_dense = units_dense
        self.num_layers = num_layers
        self.task = task
        self.loss = loss
        self.pooling = pooling

        if self.task == 'regression':
            if std is not None:
                self.std = std
            else:
                raise ValueError("Standard deviation of the dataset must be "
                                 "provided for regression tasks.")
        

    def get_model(self) -> Model:
        """Retrive the model with the specified params.

        Returns:
        --------
        model: keras.models.Model
            The compiled model.
        """
        atoms = Input(name='atom_inputs', shape=(self.num_atoms, self.num_feats))
        adjms = Input(name='adjm_inputs', shape=(self.num_atoms, self.num_atoms))
        dists = Input(name='coor_inputs', shape=(self.num_atoms, self.num_atoms, 3))

        sc, vc = GraphEmbed()([atoms, dists])

        for _ in range(self.num_layers):
            sc_s = GraphSToS(self.units_conv, activation='relu')(sc)
            sc_v = GraphVToS(self.units_conv, activation='relu')([vc, dists])

            vc_s = GraphSToV(self.units_conv, activation='tanh')([sc, dists])
            vc_v = GraphVToV(self.units_conv, activation='tanh')(vc)

            sc = GraphConvS(self.units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
            vc = GraphConvV(self.units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc, vc = GraphGather(pooling=self.pooling)([sc, vc])
        sc_out = Dense(self.units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
        sc_out = Dense(self.units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

        vc_out = TimeDistributed(Dense(self.units_dense, activation='relu', 
                                       kernel_regularizer=l2(0.005)))(vc)
        vc_out = TimeDistributed(Dense(self.units_dense, activation='relu', 
                                       kernel_regularizer=l2(0.005)))(vc_out)
        vc_out = Flatten()(vc_out)

        out = Concatenate(axis=-1)([sc_out, vc_out])

        if self.task == "regression":
            out = Dense(self.num_outputs, activation='linear', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), 
                          loss=self.loss, 
                          metrics=[std_mae(std=self.std), 
                                   std_rmse(std=self.std), 
                                   std_r2(std=self.std)])
        elif self.task == "binary":
            out = Dense(self.num_outputs, activation='sigmoid', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), loss=self.loss)
        elif self.task == "classification":
            out = Dense(self.num_outputs, activation='softmax', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), loss=self.loss)
        else:
            raise ValueError("Unsupported task on model generation.")
        
        return model
        