# {Se}curity {Fi}ndings {De}duplication {F}ramework

This repository contains code written to achieve the goals of my Master Thesis. It uses NLP semantic similarity techniques to identify duplicates in DevOps security findings reports, and compares the results with ground truth data. This would enable identification of the most suitable technique for security findings deduplication. The terminologies "identification of duplicates" and "deduplication" are used interchangeably in code and documentation, since latter is the goal of former.

The code contains the following subpackages:
- `runcases`: contain runnable deduplication experiments e.g., deduplicating reports from all static tools
- `techniques`: wrap existing implementations of NLP techniques for modularity within the application
- `dataloaders`: contain modules used to load and provide data in the format compatible with implemented `techniques`
- `corpus_formats`: contain data structures to help `dataloaders` extract the right information from sources

The data from security tool reports is aggregated and stored in a dataset, available in the `datasets` folder. These datasets were created using [SeFiLa](https://github.com/abdullahgulraiz/SeFiLa).

## Getting started
The code uses [Pipenv](https://pypi.org/project/pipenv/) for dependency management. Please make sure its installed. Clone the source code from repository and change to directory:

```
git clone https://github.com/abdullahgulraiz/SeFiDeF.git
cd SeFiDeF
```
Install all dependencies:

```
pipenv install
```

The entrypoint of application is `main.py`. To execute the runcases contained therein, please run:

```
pipenv run python main.py
```
