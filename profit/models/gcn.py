
from keras.layers import Add, Concatenate, Dense, Dropout, Flatten, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from profit.layers.convs import GraphConvS, GraphConvV
from profit.layers.embed import GraphEmbed
from profit.layers.gather import GraphGather
from profit.layers.nodes import GraphSToS, GraphSToV, GraphVToS, GraphVToV
from profit.utils.training.loss import std_mae, std_rmse, std_r2

from typing import Optional


class GCN(object):
    """Graph Convolution Network.

    Num of outputs can be automatically inferred from the label names, if the task is multitask or not. 
    
    task type can be automatically determined from the type of data. If it is a float or int...classification or regression.
    Should this be handled by this, or by the trainer class?????? Well, if its handled in this, it has to be determined by every single model, which means it needs to be impemented over and over, which is wrong. Rather, 
    """

    def __init__(self, 
                 num_atoms: int, 
                 num_feats: int, 
                 num_outputs: int,
                 units_conv: Optional[int]=128, 
                 units_dense: Optional[int]=128, 
                 num_layers: Optional[int]=2, 
                 task: Optional[str]='regression',
                 loss: Optional[str]="mse", 
                 pooling: Optional[str]="max", 
                 std: Optional[float]=1.0):
        self.num_atoms = num_atoms
        self.num_feats = num_feats
        self.num_outputs = num_outputs
        self.units_conv = units_conv
        self.units_dense = units_dense
        self.num_layers = num_layers
        self.task = task
        self.loss = loss
        self.pooling = pooling
        self.std = std
        

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
            model.compile(optimizer=Adam(lr=0.001), loss=self.loss, 
                          metrics=[std_mae(std=self.std), std_rmse(std=self.std)])
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
