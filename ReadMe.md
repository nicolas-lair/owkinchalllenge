### Requirement

You need the following libraries to run the code

```bash
pytorch
numpy
tqdm
pandas
scikit-learn
```

### Organisation

#### Base code
- `model.py` contains the models definitions
- `loader.py` defines the Dataset structures and a dataloader
- `utils.py` defines a set of useful functions (training, evaluation, loading etc.)

#### Config
The configurations params are in `config.py`. 

#### Run code
- `trainer.py` allows to train a model and compute the predictions on the test set
- `crossval.py` allows to cross validate a model
To choose the model to train or cross-validate, just call the right config class in the head of the file.

