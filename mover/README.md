# MOVER

MOVER: Multi-Objective Optimization for VErsatile Materials Research is an
implementation of Multi-objective Optimization for Materials Discovery via AdaptiveDesign

|             |                                                                                                                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| release     | ![GitHub release](https://img.shields.io/github/v/release/Feugmo-Group/mover?display_name=tag&include_prereleases)![](https://img.shields.io/badge/Maintained-Yes-indigo)                                                      |
| Maintainers | ![Maintainers](https://img.shields.io/badge/maintainers-3-success.svg "Number of maintainers")![](https://img.shields.io/github/contributors/Feugmo-Group/mover)![](https://img.shields.io/badge/Contributions-Accepting-pink) |
| Activity    | ![](https://img.shields.io/badge/Pull_Requests-Accepting-yellow)![](https://img.shields.io/github/forks/Feugmo-Group/mover)![](https://img.shields.io/github/issues/Feugmo-Group/mover)                                        |
| Stats       | ![GitHub stars](https://img.shields.io/github/stars/Feugmo-Group/mover)                                                                                                                                                        |
| LICENSE     | ![GitHub license](https://img.shields.io/github/license/Feugmo-Group/mover)                                                                                                                                                    |

## Features

- **Active learning:** : The approach is based on the research paper: Gopakumar, A. M., Balachandran, P. V., Xue, D., Gubernatis, J. E., & Lookman, T. (2018). Multi-objective Optimization for Materials Discovery via Adaptive Design. Scientific Reports, 8(1), 1-12. https://doi.org/10.1038/s41598-018-21936-3
- **reinforcement learning:** The approach is based on the research paper: Liu, Q.; Cui, C.; Fan, Q. Self-Adaptive Constrained Multi-Objective Differential Evolution Algorithm Based on the State–Action–Reward–State–Action Method. Mathematics 2022, 10, 813. https://doi.org/10.3390/math10050813

<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Install poetry (https://github.com/python-poetry/poetry)

   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   poetry self update
   poetry config --list # please check the "virtualenvs.path"
   ```

2. Clone the repo

   ```sh
   git clone https://github.com/Feugmo-Group/mover.git
   cd mover
   ```

3. Create a Virtual Environment

   ```sh
   brew install pyenv
   pyenv install 3.x.x
   pyenv local 3.x.x  # Activate Python 3.x.x for the current project
   poetry config virtualenvs.create
   ```

4. Install the packages
   ```sh
   poetry install
   poetry check
   poetry run pytest
   poetry build
   ```
5. Listing the current configuration
   ```sh
   poetry config --list
   ```

<!-- USAGE EXAMPLES -->

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.
You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project

2. Create your Feature Branch
   ```sh
    git branch BRANCH-NAME
    git checkout BRANCH-NAME
   ```
3. Install poetry (https://github.com/python-poetry/poetry)

   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   poetry self update
   mkdir $HOME/.venv
   poetry  config virtualenvs.path $HOME/.venv
   poetry config --list # please check the "virtualenvs.path"
   ```

4. Install pre-commit (https://pre-commit.com)

   ```sh
   brew install pre-commit
   or pip install pre-commit
   ```

5. Activate pre-commit and update

   ```sh
   pre-commit install
   pre-commit autoupdate
   ```

6. If you want to only lint the changes to files (for example, if you’re incrementally linting/formatting files rather than in One Big Commit)

   ```sh
   pre-commit run --from-ref $(git merge-base ${TARGET_BRANCH}) --to-ref HEAD
   ```

7. Commit your Changes
   ```sh
   poetry check
   poetry lock --no-update
   pre-commit run
   git git add -u
   git commit -m "a short description of the change"
   ```
8. Push to the Branch
   ```sh
   git push
   ```
9. Open a Pull Request

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

## Contact

Conrard Tetsassi - [@FeugmoC](https://twitter.com/FeugmoC) - cgtetsas@uwaterloo.ca

Project Link: [https://github.com/Feugmo-Group/ElectroVault](https://github.com/Feugmo-Group/ElectroVault)

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

- []()
- []()
