# Audinter

Interpretability and Explainability metrics for ML model auditing.

## Installation

```bash
$ pip install audinter
```

## Usage

```python
from audinter.metrics import algorithm_class_score
from audinter.metrics import correlated_features_score
from audinter.metrics import model_size
from audinter.metrics import feature_importance_score
from audinter.metrics import cv_shap_score
from audinter.metrics import all_metrics
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`audinter` was created by Antónia Brito. It is licensed under the terms of the MIT license.

## Credits

`audinter` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

The `audinter` is based on the metrics from these papers:

Sánchez, P. M. S., Celdrán, A. H., Xie, N., Bovet, G., Pérez, G. M., & Stiller, B. (2024). Federatedtrust: A solution for trustworthy federated learning. Future Generation Computer Systems, 152, 83-98.

Huertas Celdran, A., Kreischer, J., Demirci, M., Leupp, J., Sánchez Sánchez, P. M., Figueredo Franco, M., & Stiller, B. (2023, February). A framework quantifying trustworthiness of supervised machine and deep learning models. In CEUR Workshop Proceedings (No. 3381, pp. 1-14). CEUR-WS.

## Funding information

Agenda “Center for Responsible AI”, nr. C645008882-00000055, investment project nr. 62, financed by the Recovery and Resilience Plan (PRR) and by European Union -  NextGeneration EU.

AISym4Med (101095387) supported by Horizon Europe Cluster 1: Health, ConnectedHealth (n.o 46858), supported by Competitiveness and Internationalisation Operational Programme (POCI) and Lisbon Regional Operational Programme (LISBOA 2020), under the PORTUGAL 2020 Partnership Agreement, through the European Regional Development Fund (ERDF)