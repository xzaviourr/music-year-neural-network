# From-Scratch Neural Network for Music Release-Year Prediction

An IIT Bombay CS 725 academic project that implements feed-forward neural
networks with NumPy rather than a deep-learning framework. The primary task is
regression from 90 timbre-derived audio features to a song's release year; a
four-class release-era classifier is included as an extension.

> **Historical academic project:** this repository is archived and is not
> maintained. The results are assignment-specific and should not be interpreted
> as general music-dating accuracy.

## Implementation

The code implements:

- dense layers with weights and biases;
- ReLU hidden activations;
- forward propagation and backpropagation;
- mini-batch gradient descent;
- mean-squared error for regression;
- softmax/cross-entropy classification;
- Min-Max scaling; and
- PCA-based feature reduction experiments.

`complete_code/` contains the assignment deliverables, while
`regression_main.py` is a consolidated regression workflow.

## Data

Each row has 90 timbre features: 12 timbre-average values and 78
timbre-covariance values. Regression labels span release years 1922–2011.
The classification extension groups songs into Very Old, Old, New, and Recent.

The assignment identifies the data as a course-prepared version of the
[UCI YearPredictionMSD dataset](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd),
derived from the Million Song Dataset. The checked-in train/dev/test splits
remain subject to the source and course terms; their subset construction and
sampling are not documented here.

## Setup and use

```bash
git clone https://github.com/xzaviourr/music-year-neural-network.git
cd music-year-neural-network
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python regression_main.py
```

The scripts assume their original working-directory layout and may train for a
long time. Review file paths and hyperparameters before running. Historical
notebook output is available in `regression_predictions.ipynb`.

## Reported results

The final commit records regression RMSE of:

- **10.60 years** on the training split; and
- **10.87 years** on the held-out test split.

Saved plots compare train/dev loss for batch sizes 32 and 64. These metrics
come from one course split and implementation; no repeated trials, confidence
intervals, or external validation are provided.

## Limitations

- Release year is only weakly determined by timbre.
- Dataset age, genre, geography, and collection biases can limit
  generalization.
- The repository includes assignment data and instructions, not a reusable
  package or stable command-line interface.
- Reproducing exact results can depend on legacy library versions and the
  original data split.

No license is asserted because the repository incorporates course materials
and third-party dataset files whose terms are separate from the original code.
