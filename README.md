# Effects of Higher-Order Interactions and Homophily on Information Access Inequality

This repository contains code for the paper *Effects of Higher-Order Interactions and Homophily on Information Access Inequality*.

We provide a custom hypergraph class `HyperGraph` that provides access to basic hypergraph properties and inequality and a child
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

## 📂 Repository Structure

### Interactive Tutorial
- `tutorial.ipynb`: Step-by-step walkthrough of the framework, including:
  - Generating synthetic hypergraphs
  - Cleaning and converting real-world data
  - Running SI simulations
  - Measuring information access inequality, acquisition fairness, and diffusion fairness

### Core Classes and Utilities
- `HyperGraph.py`: Defines the `HyperGraph` class and its extension `HyperGraphAsymmetricSI`, an environment for a non-linear Susceptible-Infected (SI) model on hypergraphs.
- `HyperGraphHelper.py`: Helper functions for hypergraph construction, simulation setup, and fairness evaluation.
- `fairness_measures.py`: Implements acquisition fairness and diffusion fairness metrics from Zappalà et al. (2024).
- `stats.py`: Extracts and stores descriptive statistics from hypergraphs.

### Simulation Pipeline
- `generate_params.py`: Generates `.json` parameter files to configure experiments on synthetic or real-world hypergraphs.
- `simulation.py`: Runs the full experiment pipeline (hypergraph generation + simulation + output) using configuration files.

### Plotting Scripts
- `plot_scripts/`: Contains all plotting code used to generate the figures in the paper.

### Real-World Data Processing
- `real_world.py`: Cleans and processes real-world datasets, gender-labels nodes, generates hypergraphs, and extracts summary statistics.

Dummy versions of select datasets are provided:
```
data/real_world/
├── dummy_hospital/
├── dummy_highschool/
```

Real-world datasets processed in this repo include:
- **Hospital** [Vanhems et al., 2013]
  - Go to SocioPatterns ([here](http://www.sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/ )) and (1) download `detailed_list_of_contacts_Hospital.dat_.gz`, unzip it, and rename it to `hospital.csv`.
- **High School** [Mastrandrea et al., 2015; Benson et al., 2018]
  - Go to SocioPatterns ([here](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/)) and (1) download `High-School_data_2013.csv.gz`, unzip it, and rename it to `highschool.csv`, and (2) copy the data from `Metadata, tab separated` into a file titled `metadata.csv`.
- **Primary School** [Stehlé et al., 2011]
  - Go to SocioPatterns ([here](http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/)) and (1) download `primaryschool.csv.gz`, unzip it, and rename it to `primaryschool.csv`, and (2) copy the data from `Metadata, tab separated` into a file titled `metadata.csv`.
- **Senate Bills** [Fowler, 2006; Chodrow et al., 2021]
  - Go to [https://www.cs.cornell.edu/~arb/data/senate-bills/](https://www.cs.cornell.edu/~arb/data/senate-bills/) and download `senate-bills.zip`. Unzip the file and rename `hyperedges-senate-bills.txt` to `senatebills.csv`.
- **House Bills** [Fowler, 2006; Chodrow et al., 2021]
  - Go to [https://www.cs.cornell.edu/~arb/data/house-bills/](https://www.cs.cornell.edu/~arb/data/house-bills/) and download `house-bills.zip`. Unzip the file and rename `hyperedges-house-bills.txt` to `housebills.csv`.
- **DBLP Computer Science Collaboration** [dblp.org, 2024]
  - Go to (dblp.org)[https://dblp.org/] and download `dblp.xml.gz` and `dblp.dtd`. Parse the .xml file to generate co-authorship lists, and save the file with one list of co-authors per row as `dblp.csv`.
- **APS Physics Collaboration** [https://journals.aps.org/datasets, 2024]
  - Request access to the APS data from (https://journals.aps.org/datasets)[https://journals.aps.org/datasets]. Then, use the APS script in this repository to generate the `aps.csv` and `metadata.csv` files.

#### File Structure and Expected Inputs

After downloading and renaming the data as explained above, ensure that the following files are present:
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

Each dataset must include the appropriate metadata and either edge lists or hyperedge definitions as shown. You can then use the script `real_world.py` as shown in `tutorial.ipynb` to generate real-world hypergraphs and simulate contagion over the hypergraphs.

## Citations

- Zappalà, C., Gallo, L., Bachmann, J., Battiston, F. & Karimi, F. Gender disparities in teh dissemination and acquisition of scientific knowledge, DOI: [10.48550/arXiv.2407.17441](https://arxiv.org/abs/2407.17441) (2025). [2407.17441](https://arxiv.org/abs/2407.17441).
- Vanhems, P. _et al._ Estimating potential infection transmission routes in hospital wards using wearable proximity sensors. _PloS one_ **8**, e73970 (2013).
- Benson, A. R., Abebe, R., Schaub, M. T., Jadbabaie, A. & Kleinberg, J. Simplicial closure and higher-order link prediction. _Proc. Natl. Acad. Sci._ DOI: [10.1073/pnas.1800683115](https://www.pnas.org/doi/10.1073/pnas.1800683115) (2018).
- Mastrandrea, R., Fournet, J. & Barrat, A. Contact patterns in high school: A comparison between data collected using wearable sensors, contact diaries, and friendship surveys. _PloS one_ **10**, e0136497, DOI: [10.1371/journal.pone.0136497](https://pubmed.ncbi.nlm.nih.gov/26325289/) (2015).
- Chodrow, P. S., Veldt, N. & Benson, A. R. Generative hypergraph clustering: From blockmodels to modularity. _Sci. Adv._ **7**, eabh1303, DOI: [10.1126/sciadv.abh1303](https://www.science.org/doi/10.1126/sciadv.abh1303) (2021).
- Stehlé, J. _et al._ High-resolution measurements of face-to-face contact patterns in a primary school. _PloS one_ **6**, e23176, DOI: [10.1371/journal.pone.0023176](https://pubmed.ncbi.nlm.nih.gov/21858018/) (2011).
- Fowler, J. H. Connecting the congress: A study of cosponsorship networks. _Polit. Analysis_ **14**, 456-487, DOI: [10.1093/pan/mpl002](https://www.cambridge.org/core/journals/political-analysis/article/abs/connecting-the-congress-a-study-of-cosponsorship-networks/B42907E13C3D1F12BBC7618C8E0EECED) (2006).
- Fowler, J. H. Legislative cosponsorship networks in the US house and senate. _Soc. Networks_ **28**, 454-465, DOI: [10.1016/j.socnet.2005.11.003](https://pdodds.w3.uvm.edu/files/papers/others/2006/fowler2006b.pdf) (2006).
- DBLP. DBLP computer science bibliography (2024). [https://dblp.org/xml/](https://dblp.org/xml/), (2024-10-11 release).
- American Physical Society (APS) Citation Dataset (2024). [https://journals.aps.org/](https://journals.aps.org/) datasets (2024-11-15 release). 
