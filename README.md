<p align="center">
  <img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

**M**edical **O**pen **N**etwork for **AI**

![Supported Python versions](https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/python.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/monai.svg)](https://badge.fury.io/py/monai)
[![docker](https://img.shields.io/badge/docker-pull-green.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/projectmonai/monai)
[![conda](https://img.shields.io/conda/vn/conda-forge/monai?color=green)](https://anaconda.org/conda-forge/monai)

[![premerge](https://github.com/Project-MONAI/MONAI/actions/workflows/pythonapp.yml/badge.svg?branch=dev)](https://github.com/Project-MONAI/MONAI/actions/workflows/pythonapp.yml)
[![postmerge](https://img.shields.io/github/checks-status/project-monai/monai/dev?label=postmerge)](https://github.com/Project-MONAI/MONAI/actions?query=branch%3Adev)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://docs.monai.io/en/latest/)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAI/branch/dev/graph/badge.svg?token=6FTC7U1JJ4)](https://codecov.io/gh/Project-MONAI/MONAI)
[![monai Downloads Last Month](https://assets.piptrends.com/get-last-month-downloads-badge/monai.svg 'monai Downloads Last Month by pip Trends')](https://piptrends.com/package/monai)

MONAI is a [PyTorch](https://pytorch.org/)-based, [open-source](https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE) framework for deep learning in healthcare imaging, part of the [PyTorch Ecosystem](https://pytorch.org/ecosystem/).
Its ambitions are as follows:
- Developing a community of academic, industrial and clinical researchers collaborating on a common foundation;
- Creating state-of-the-art, end-to-end training workflows for healthcare imaging;
- Providing researchers with the optimized and standardized way to create and evaluate deep learning models.


## Features
> _Please see [the technical highlights](https://docs.monai.io/en/latest/highlights.html) and [What's New](https://docs.monai.io/en/latest/whatsnew.html) of the milestone releases._

- flexible pre-processing for multi-dimensional medical imaging data;
- compositional & portable APIs for ease of integration in existing workflows;
- domain-specific implementations for networks, losses, evaluation metrics and more;
- customizable design for varying user expertise;
- multi-GPU multi-node data parallelism support.


## Installation

To install [the current release](https://pypi.org/project/monai/), you can simply run:

```bash
pip install monai
```

Please refer to [the installation guide](https://docs.monai.io/en/latest/installation.html) for other installation options.

## Getting Started

[MedNIST demo](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe) and [MONAI for PyTorch Users](https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T) are available on Colab.

Examples and notebook tutorials are located at [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials).

Technical documentation is available at [docs.monai.io](https://docs.monai.io).

## Citation

If you have used MONAI in your research, please cite us! The citation can be exported from: https://arxiv.org/abs/2211.02701.

## Model Zoo
[The MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) is a place for researchers and data scientists to share the latest and great models from the community.
Utilizing [the MONAI Bundle format](https://docs.monai.io/en/latest/bundle_intro.html) makes it easy to [get started](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo) building workflows with MONAI.

## Quick Start Example
Here's a simple example of using MONAI for medical image preprocessing:

```python
import monai
from monai.transforms import LoadImage, Spacing

# Load a medical image
image = LoadImage()(image_path)
# Resample the image to a specific spacing
spaced_image = Spacing(pixdim=(1.5, 1.5, 2.0))(image)
```

## Contributing
For guidance on making a contribution to MONAI, see the [contributing guidelines](https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md).

## Community
Join the conversation on Twitter/X [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over on [MONAI's GitHub Discussions tab](https://github.com/Project-MONAI/MONAI/discussions).

## Links
- Website: https://monai.io/
- API documentation (milestone): https://docs.monai.io/
- API documentation (latest dev): https://docs.monai.io/en/latest/
- Code: https://github.com/Project-MONAI/MONAI
- Project tracker: https://github.com/Project-MONAI/MONAI/projects
- Issue tracker: https://github.com/Project-MONAI/MONAI/issues
- Wiki: https://github.com/Project-MONAI/MONAI/wiki
- Test status: https://github.com/Project-MONAI/MONAI/actions
- PyPI package: https://pypi.org/project/monai/
- conda-forge: https://anaconda.org/conda-forge/monai
- Weekly previews: https://pypi.org/project/monai-weekly/
- Docker Hub: https://hub.docker.com/r/projectmonai/monai
