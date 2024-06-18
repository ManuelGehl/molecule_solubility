# molecule_solubility

This project is based on the AqSolDB dataset, which is a curated dataset of chemical compounds with their aqueous solubility and several 2-dimensional descriptors. The data was received from the Harvard Dataverse on December 11, 2023 (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8).

## Dataset Overview

The dataset comprises 9982 entries with 26 columns, including various chemical descriptors. 
Notably, the dataset contains no missing values or duplicate entries, although the "Name" column contains 89 duplicate entries due to the use of the same name for different isomers. Since the number of duplicate names is relatively small, half of them were randomly removed.

Solubility values are presented as log values of the measured solubility in mol/l (LogS) and range from -13 to 2, with a mean of -2.89 and a standard deviation of 2.37. The values are concentrated between -6 and 2, peak at -2, and have a shoulder between -8 and -6. According to the referenced paper, solubility is categorized as highly soluble (LogS >= 0), soluble (0 > LogS >= -2), slightly soluble (-2 > LogS >= -4), and insoluble (LogS < -4). 

Molecular weight (MolWt) has a very high range of 9 - 5300 g/mol with a mean of 266 g/mol, showing that very large molecules (above 1000 Da) skew the data set, so filtering out these outliers improves data presentation.

The correlation analysis showed that the most important descriptors are MolLogP (0.61), MolMR (0.42), MolWt (0.37), HeavyAtomCount (0.35), LabuteASA (0.35), NumValenceElectrons (0.35), NumAromaticRings (0.34) and RingCount (0.33). Since the participation coefficient MolLogP describes the hydrophobicity of a compound, it also strongly influences the solubility.

Most high molecular weight compounds affect many descriptors but not necessarily solubility. However, since these high molecular weight compounds are rather rare in the data set, an attempt was made to exclude them for the sake of a better representation. Compounds with molecular weight values with z-scores greater than 4 standard deviations from the mean were considered outliers. These were 85, with the smallest compound having a molecular weight of 1006.7 Da. For simplicity, a cutoff of 1000 Da was introduced into the data set.

## Data preprocessing and feature engineering

Based on the data analysis described above, several preprocessing steps were performed:

- Remove duplicate entries with the same name value
- Remove molecules heavier than 1000 Da
- Categorized descriptors such as RingCount, NumAromaticRings into binary features describing whether compounds have these rings at all or not
- Categorize solubility based on paper categories
- Scale continuous features using standard scaling

## Model building



