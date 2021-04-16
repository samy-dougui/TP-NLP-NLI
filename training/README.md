## Training

This folder is an implementation of a pytorch_lightning class.

It trains the models as in the jupyter notebook.
But pytorch_lightning allows to simplify it, and to easily log more information. For example, the equivalent of the training function in the jupyter notebook tp_8.ipynb is the training_step function inside the model SentencesClassification.

During this training, we used the F1-score and the accuracy.
All the logs can be displayed with tensorboard using the following command:
`tensorboard --logdir logs/default`
