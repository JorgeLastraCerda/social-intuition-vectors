# NSF-Discrimination

This repository contains data and code for a meta-analysis investigating North American labor market discrimination. 

## Citation
If you use our data and/or code, please cite our paper:

```bibtex
@article{gallo2024perceived,
  title={Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments},
  author={Gallo, Marcos and Hausladen, Carina I and Hsu, Ming and Jenkins, Adrianna C and Ona, Vaida and Camerer, Colin F},
  journal={Plos one},
  volume={19},
  number={7},
  pages={e0304723},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

Direct Link to Preprint: [Social Perceptions of Warmth and Competence Predict Callback Rates in North American Labor Market Experiments](https://doi.org/10.1371/journal.pone.0304723)


# File Structure

The repository contains the following key directories and files:

- `0_data/`: This directory contains data used in the analysis.
  - `extracted_data`: Data points extracted from published correspondence studies. Two independent researchers cross-checked all extractions.
  - `published_data`: .txt files with links to public repositories where authors store raw datasets from published correspondence studies. We use these data sets in our study.
  - `ratings`: Warmth and competence ratings collected via Prolific for names and categories.
- `book/`: This directory contains the R code. It includes several .qmd files that are compiled into a single Quarto book.
- `requirements.txt`: This file specifies the exact versions of the R packages used in the project.

# Replication Instructions

Replicating our paper is straightforward! The Quarto book contains all figures and tables included in the paper. Additionally, all numbers in the text are dynamic, so it can be easily understood in which object the number is stored and how it was calculated.
