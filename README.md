This repository acts as a lite version for the World Map Representation work done by Core Park (found here: https://github.com/cfpark00/world-map-representation)

This goal of this repository is to create a template that can be used to investigate the following:

How internal representations emergy across different tasks, task families, model architectures, and training regimes.

To accomplish this, the Lite World Map Representation (LWMap) was introduced.

CURRENT STEPS TO USE (as of 2026-07-07)

```
git clone https://github.com/trhillis/worldMapLite.git
```
OR
```
git clone git@github.com:trhillis/worldMapLite.git
```

```
cd worldMapLite
pip install -r requirements.txt

python train.py
```

This will train a model to predict the distance between two locations on an XY plane.

Currently, the locations are fake and created within the code.

To view the results:
```
python analysis.py
```

Currently the repository behaves as such:

```
Synthetic World
        ↓
Distance Dataset
        ↓
Transformer Training
        ↓
Hidden Representations
        ↓
PCA Visualization
```
In words: it trains a hidden transformer on synthetic geometry data using train.py, then using analysis.py checks whether its hidden states start to look like a map.

Currently the model successfully learns the prediction task, however a map does not yet emerge. Currently, we observe an amorphous cloud instead of any kind of latent manifold. 

RESEARCH GOALS:
```
Use this repository to answer some of the following questions:
- When do latent representations emerge?
- What kinds of tasks produce them?
- Which tasks fail to produce them?
- How much supervision is necessary?
- Are there phase transitions?
- What happens if the underlying world is not Euclidean?
- Do independently trained models converge?
- Can the representation be destroyed or repaired?
```

The goal is to understand what conditions lead to the emergence of structured latent representations.