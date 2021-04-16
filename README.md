# TP-NLP-NLI

# Explanation of the repository

This Github repository is divided in different parts:

- The notebook, which contains the explanation of the models and the choices we've made, an example of how to use the trained models and the visualization of the attention (**the visualization is only working on a _Jupyter Notebook_** )
- The server folder, which is a small containerized Flask-API that allows a user to test the trained models from the command line or an API platform (for example Postman). More on that on the README.md inside this folder
- The training folder that contains a main.py and a model.py that are an implementation of the training/testing/validating of a model using [Pytorch-Lightning](https://www.pytorchlightning.ai). It also contains logs for tensorboard (more on this in the readme of the folder).

## pytorch_lightning related links

https://github.com/PyTorchLightning/pytorch-lightning/issues/2171
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-epoch-level-metrics
https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#under-the-hood
