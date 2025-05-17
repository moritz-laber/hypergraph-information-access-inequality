# Effects of Higher-Order Interactions and Homophily on Information Access Inequality

This repository contains code for the paper *Effects of Higher-Order Interactions and Homophily on Information Access Inequality*.

We provide a custom hypergraph class `HyperGraph`that provides access to basic hypergraph properties and inequality. And a child
class `HyperGraphAsymmetricSI` for simulations of a non-linear, group-asymmetric Suscpeptible-Infected model.

We also provide code to generate hypergraphs from our hypergraph model, `CSCM`. 

Finally, we provide scripts that were used to preprocess to conduct simulations, preprocess results, and create plots for paper.

If you found this code useful for your own research, please cite our paper.

```
@article{laber2025,
  title = {Effects of Higher-Order Interactions and Homophily on Information Access Inequality},
  author = {Laber, Moritz and Dies, Samantha and Ehlert, Joseph and Klein, Brennan and Eliassi-Rad, Tina},
  year = {2025},
  doi = {doi},
}
````



## Real-World Hypergraph Generation Pipeline

Here is the pipeline used to processes real-world datasets to generate hypergraphs, infer gender labels, and compute summary statistics.

It supports datasets from human contact networks, legislative cosponsorship networks, and academic co-authorship networks.

### File Structure and Expected Inputs

Before running the pipeline, ensure that the following files are present:

```
data/real_world/
├── highschool/
│   └── edges.csv
├── primaryschool/
│   └── edges.csv
├── hospital/
│   └── edges.csv
├── housebills/
│   ├── node-names-house-bills.txt
│   └── node-labels-house-bills.txt
├── senatebills/
│   ├── node-names-senate-bills.txt
│   └── node-labels-senate-bills.txt
├── dblp/
│   ├── author.csv
│   └── edge_list.csv
├── aps/
│   ├── author.csv
│   └── edge_list.csv
```

Each dataset must include the appropriate metadata and either edge lists or hyperedge definitions as shown.

### Running the Pipeline

To execute the full pipeline, run:

```bash
python real_world.py
```

This will:

1. Clean and preprocess data (reindex nodes, compute LCCs)
2. Run gender labeling using GenderAPI and Genderize.io
3. Generate hypergraph objects
4. Compute and save summary statistics

You may comment out steps in `main()` to skip parts of the pipeline.

### Required API Keys

Gender labeling uses:
- [GenderAPI](https://gender-api.com/)
- [Genderize.io](https://genderize.io/)

To enable gender inference, edit `real_world.py` and set:

```python
GENDER_API_KEY = "your_genderapi_key"
GENDERIZE_IO_KEY = "your_genderize_io_key"
```

If no keys are provided, gender labeling will be skipped.

### Output Files

The pipeline generates:

```
data/real_world/
├── <dataset>/
│   ├── metadata.csv              # Cleaned and labeled node data
│   ├── <dataset>.csv             # Reindexed edge or hyperedge list
│   ├── hypergraphs/
│   │   └── <dataset>_lcc_hg.pkl  # Hypergraph object for LCC
├── <dataset>gender/              # For datasets with probabilistic gender labeling
│   ├── metadata.csv
│   ├── hypergraphs_genderapi/
│   │   └── hg_lcc_*.pkl
│   └── hypergraphs_genderizerio/
│       └── hg_lcc_*.pkl
├── real_hg_info.csv              # Summary statistics for standard hypergraphs
├── real_hg_pred_gender_info.csv # Summary statistics for gender-labeled hypergraphs
```

### Summary Statistics

For each dataset, the following statistics are computed:
- Number of nodes and edges
- Minority group proportion (`p_m`)
- Average degree overall, for majority, and for minority groups
- Power inequality
- Moment glass ceiling

### Real-Worlk Data Source Citations

#### Hospital
- Vanhems et al. (2013). *Estimating potential infection transmission routes in hospital wards using wearable proximity sensors*. PLoS ONE.

#### High School
- Benson et al. (2018). *Simplicial closure and higher-order link prediction*. PNAS.
- Mastrandrea et al. (2015). *Contact patterns in a high school*. PLOS ONE.

#### Primary School
- Chodrow et al. (2021). *Hypergraph clustering: from blockmodels to modularity*. Science Advances.
- Stehlé et al. (2011). *Face-to-face contact patterns in a primary school*. PLoS ONE.

#### House and Senate Bills
- Chodrow et al. (2021). *Hypergraph clustering: from blockmodels to modularity*. Science Advances.
- Fowler (2006). *Connecting the congress: A study of cosponsorship networks*. Political Analysis.
- Fowler (2006). *Legislative cosponsorship networks in the US house and senate*. Social Networks.

#### DBLP
- DBLP Computer Science Bibliography. [https://dblp.org/](https://dblp.org/)
