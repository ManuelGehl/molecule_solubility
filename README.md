# molecule_solubility

This project is based on the AqSolDB dataset, which is a curated dataset of 9982 chemical compounds with their aqueous solubility and several 2-dimensional descriptors. The data was received from the Harvard Dataverse on December 11, 2023 (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8).

Data preprocessing involved removing duplicate names and high molecular weight outliers, categorizing descriptors, and scaling continuous features. Feature importance analysis identified key descriptors, with MolLogP being the most influential. Multiple regression models were screened and fine-tuned, with Support Vector Regressor, Gradient Boosting, k-Nearest Neighbors, and Multi-Layer Perceptron showing the best performance. Ensemble models combining these regressors outperformed individual models, with the stacking regressor achieving the lowest mean absolute error (MAE). Error analysis indicated that the model is most accurate for compounds with solubility around -2, but predictions deviate for higher and lower solubilities, suggesting further improvements could be made with a more uniform dataset or specialized models for different solubility ranges.

## Dataset Overview

The dataset comprises 9982 entries with 26 columns, including various chemical descriptors. 
Notably, the dataset contains no missing values or duplicate entries, although the "Name" column contains 89 duplicate entries due to the use of the same name for different isomers. Since the number of duplicate names is relatively small, half of them were randomly removed.

Solubility values are presented as log values of the measured solubility in mol/l (LogS) and range from -13 to 2, with a mean of -2.89 and a standard deviation of 2.37. The values are concentrated between -6 and 2, peak at -2, and have a shoulder between -8 and -6. According to the referenced paper, solubility is categorized as highly soluble (LogS >= 0), soluble (0 > LogS >= -2), slightly soluble (-2 > LogS >= -4), and insoluble (LogS < -4) (**Fig. 1**).

<br></br>
<img src=figures/solubility_distribution.png>

**Figure 1: Histogram of logarithmic solubility values for the entire data set.**
<br></br>

Molecular weight (MolWt) has a very high range of 9 - 5300 g/mol with a mean of 266 g/mol, showing that very large molecules (above 1000 Da) skew the data set, so filtering out these outliers improves data presentation.

The correlation analysis showed that the most important descriptors are MolLogP (0.61), MolMR (0.42), MolWt (0.37), HeavyAtomCount (0.35), LabuteASA (0.35), NumValenceElectrons (0.35), NumAromaticRings (0.34) and RingCount (0.33). Since the participation coefficient MolLogP describes the hydrophobicity of a compound, it also strongly influences the solubility.

<br></br>
<img src=figures/corr_matrix.png>

**Figure 2: Correlation matrix of all numeric features in the data set.**
<br></br>

Most high molecular weight compounds affect many descriptors but not necessarily solubility. However, since these high molecular weight compounds are rather rare in the data set, an attempt was made to exclude them for the sake of a better representation. Compounds with molecular weight values with z-scores greater than 4 standard deviations from the mean were considered outliers. These were 85, with the smallest compound having a molecular weight of 1006.7 Da. For simplicity, a cutoff of 1000 Da was introduced into the data set.

<br></br>
<img src=figures/outliers.png>

**Figure 3: Molecular weight plotted against solubility and outliers marked.**
<br></br>

## Data preprocessing and feature engineering

Based on the data analysis described above, several preprocessing steps were performed:

- Remove duplicate entries with the same name value
- Remove molecules heavier than 1000 Da
- Categorized descriptors such as RingCount, NumAromaticRings into binary features describing whether compounds have these rings at all or not
- Categorize solubility based on paper categories
- Scale continuous features using standard scaling

## Feature importance

Based on the data set and the engineered features, 2 strategies emerged:
1. Use continuous features to describe rings in compounds
2. Use categorical features to describe rings in compounds

Since the dataset contains many features, the first step was to identify the most important features using a Random Forest Regressor (**Fig. 4**). 

<br></br>
<img src=figures/feature_importance.png>

**Figure 4: Feature importance based on a Random Forest Regressor using continuous features.**
<br></br>

Regardless of whether the ring features were categorical or continuous, they added little to the predictive power of the models. According to the correlation analysis, the most important feature by far is the partition coefficient. Interestingly, the BertzCT is the second most important feature, which is different from the correlation matrix. The general trend is that chemical descriptors that describe accessible surface areas, branching or other more topological features are much more important than the simple atom type descriptors.

Finally, the following features were included in the training dataset used for model building:
- MolWt
- MolLogP
- MolMR
- HeavyAtomCount
- NumHDonors
- NumRotatableBonds
- NumValenceElectrons
- Labute ASA
- BertzCT

## Model screening and fine-tuning

To test different models, the default hyperparameters of the following models were used:
- Dummy Regressor
- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest
- Gradient Boosting Regressor (GB)
- k-Nearest Neighbors Regressor (KNN)
- Multi-Layer Perceptron (MLP)
- AdaBoost Regressor

The models were evaluated using 5-fold cross-validation and the mean absolute error (MAE) as a metric(**Fig. 5**). All models performed better than the dummy regressor, which gave the mean solubility for each compound. AdaBoost and Linear Regression performed slightly worse than the other models. Therefore, the SVR, GB, KNN, and MLP model types were used for fine tuning.

<br></br>
<img src=figures/model_screening.png>

**Figure 5: Performance of different models after screening with default hyperparameters.**
<br></br>

Fine-tuning was done using randomized search with a 3-fold cross-validation strategy.

## Ensemble models

The four fine-tuned models have been combined into 2 different ensembles:
1. Voting classifier using the outputs of the base models and soft voting
2. Stacking classifier using the outputs of the base models and the input features as inputs to a meta-learner (linear regression).

Both models outperformed their respective base models, but the stacking model was slightly better with a mean MAE of 0.798 +/- 0.008 compared to 0.801 +/- 0.008 for the voting model.

## Evaluate model performances on test set

On the test set, the models performed well, with the GB Regressor being the worst with an MAE of 0.858 and the MLP being the best single model with an MAE of 0.82 (**Fig. 6**). The ensembles are still the best models and the tiny gap between the two models remains with the voting regressor having an MAE of 0.803 and the stacking regressor having an MAE of 0.801.

<br></br>
<img src=figures/performance_testset.png>

**Figure 6: Performance of different models after fine-tuning on the test set.**
<br></br>

## Error analysis

For the error analysis, residuals were calculated as the difference between the predicted and true solubility of the stacking model (**Fig. 7**). The parity plot shows that most prediction-label pairs are close to the identity line, indicating accurate predictions. The model performs best for compounds with solubility around -2, reflecting the dataset's composition. However, predictions deviate more for compounds with higher or very low solubility. This suggests the model could improve with a more uniform dataset or by training separate models for different solubility ranges and combining them in an ensemble.

<br></br>
<img src=figures/parity_plot.png>

**Figure 7: Parity plot for the stacking model on the test set.**
<br></br>

## Literature

Sorkun, M.C., Khetan, A. and Er, S. (2019) ‘AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds’, Scientific Data, 6(1), p. 143. Available at: https://doi.org/10.1038/s41597-019-0151-1.
