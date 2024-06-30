# molecule_solubility

This project utilized the AqSolDB dataset, comprising 9982 chemical compounds, to predict aqueous solubility. The data was received from the Harvard Dataverse on December 11, 2023 (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8). Initial data preprocessing involved removing duplicates and high molecular weight outliers, categorizing descriptors, and scaling features. Feature importance analysis highlighted MolLogP as the most influential descriptor.

Various regression models, including Support Vector Regressor, Gradient Boosting, k-Nearest Neighbors, and Multi-Layer Perceptron, were screened and fine-tuned. Ensemble models outperformed individual models, with the stacking regressor achieving the lowest mean absolute error (MAE). Error analysis revealed that the model was most accurate for solubility around -2, with deviations for higher and lower solubility values.

The dataset was divided into organic and inorganic compounds, with only organic compounds retained for further analysis. Preprocessing steps included removing duplicates, filtering high molecular weight molecules, categorizing descriptors, counting key elements, and scaling continuous features.

The best-performing models, when tested with MACCS fingerprints and combined features, achieved an RMSE of 0.92 on the test set. Graph neural networks (GNNs) were also tested but did not outperform the models using 2D descriptors and MACCS fingerprints. Incorporating graph-level features into the GCN architecture significantly improved its performance, achieving an RMSE of 0.921.

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

RdKit was used to count the elements in each compound. Correlation analysis showed that the number of carbon atoms had the highest correlation with solubility (correlation coefficient of 0.43), similar to the correlation between MolMR and solubility. Chlorine had the second-highest correlation at 0.36, followed by sodium at 0.14. Elements Br, K, O, F, and S had correlations between 0.08 and 0.06. All other elements had lower correlations with solubility.

Using the carbon count, the dataset was divided into organic (containing carbon) and inorganic (no carbon) compounds. Only 3.3% of the compounds were inorganic. Due to the significant differences in the chemistry and solubility of organic and inorganic compounds, the inorganic compounds were excluded from the dataset.

## Data preprocessing and feature engineering

Based on the data analysis described above, several preprocessing steps were performed:

- Remove duplicate entries with the same name value
- Remove molecules heavier than 1000 Da
- Categorized descriptors such as RingCount, NumAromaticRings into binary features describing whether compounds have these rings at all or not
- Categorize solubility based on paper categories
- Count elements C, O, Na, Cl, K, Br and F and add count as feature
- Filter out anorganic compounds
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

Regardless of whether the ring features were categorical or continuous, they added little to the predictive power of the models. According to the correlation analysis, the most important feature by far is the partition coefficient. Interestingly, the BertzCT is the second most important feature, which is different from the correlation matrix. The general trend is that chemical descriptors that describe accessible surface areas, branching or other more topological features are much more important than the simple atom type descriptors. In particular, the presence of the elements oxygen, carbon and chlorine was more important than sodium, fluorine, bromine and potassium. The latter were also in the group of least important characteristics. This finding is very different to the findinds of the correlation analysis.

Finally, the following features were included in the training dataset used for model building:
- MolWt
- MolLogP
- MolMR
- HeavyAtomCount
- NumHDonors
- NumRotatableBonds
- NumValenceElectrons
- NumAromaticRings
- Labute ASA
- BertzCT
- C
- Cl
- Na
- Br
- F
- K
- O

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

The models were evaluated using 5-fold cross-validation and the root mean squared error (RMSE) as a metric (**Fig. 5**). All models performed better than the dummy regressor, which gave the mean solubility for each compound. AdaBoost and Linear Regression performed slightly worse than the other models. Therefore, the SVR, GB, KNN, and MLP model types were used for fine tuning.

<br></br>
<img src=figures/model_screening.png>

**Figure 5: Performance of different models after screening with default hyperparameters.**
<br></br>

Fine-tuning was done using randomized search with a 3-fold cross-validation strategy.

### Ensemble models

The four fine-tuned models have been combined into 2 different ensembles:
1. Voting classifier using the outputs of the base models and soft voting
2. Stacking classifier using the outputs of the base models and the input features as inputs to a meta-learner (linear regression).

Both models outperformed their respective base models with a mean RMSE of 1.04 +/- 0.03.

### Evaluate model performances on test set

On the test set, the models performed well, with the KNN Regressor being the worst with an RMSE of 1.14 and the MLP being the best single model with an RMSE of 1.04 (**Fig. 6**). The ensembles are still the best models, both resulting in an RMSE of 1.01.

<br></br>
<img src=figures/performance_testset.png>

**Figure 6: Performance of different models after fine-tuning on the test set.**
<br></br>

### Error analysis

For the error analysis, residuals were calculated as the difference between the predicted and true solubility of the voting model (**Fig. 7**). The parity plot shows that most prediction-label pairs are close to the identity line, indicating accurate predictions. The model performs best for compounds with solubility around -2, reflecting the dataset's composition. However, predictions deviate more for compounds with higher or very low solubility. This suggests the model could improve with a more uniform dataset or by training separate models for different solubility ranges and combining them in an ensemble.

<br></br>
<img src=figures/parity_plot.png>

**Figure 7: Parity plot for the stacking model on the test set.**
<br></br>

## Fingerprint Models

The chemical descriptors used so far are features that summarize different properties of molecules in one value. Morgan fingerprints, on the other hand, are designed to capture molecular features at a higher granularity by generating an embedding based on different substructures of a molecule. MACCS fingerprints are predefined substructures of molecules.

RDKit was used to generate Extended-Connectivity Fingerprints with a diameter of 4 (ECFP4) as well as MACCS fingerprints from the SMILES of the dataset. Fast model screening using default hyperparameters resulted in a slightly better RMSE for most values using MACCS fingerprints than ECFP4. Therefore, a gradient boosting (GB) regressor, a support vector regressor (SVR), and a random forest regressor were fine-tuned using MACCS as features. The mean RSME over 3 cross-folds was 1.2 for both the GB regressor and the SVR, and 1.3 for the random forest regressor.

### Models that combine descriptors and MACCS fingerprints

Both types of features, the 2D descriptors and the MACCS fingerprints, were combined. Since the best models so far for the separate feature sets were SVR, GB regressor and MLP, the focus was on them and they were used for fine-tuning. The combination of features was beneficial for model prediction, as all models had better RMSE values than the best models trained on the separate feature sets (**Tab. 1**).

**Table 1: RMSE values for 3-fold cross-validation on a dataset containing both 2D descriptors and MACCS fingerprints.**
|  Regressor | RMSE |
|----:|---------:|
|  GB | 0.98 |
| SVR | 0.95 |
| MLP | 0.98 |

Combining all three models into one voting regressor improved performance even more, reaching an RMSE of 0.93 +/- 0.03.

A final test set evaluation was performed using the best models trained on either the MACCS fingerprints alone or the combination of 2D descriptors and MACCS fingerprints (Fig. 8). The models trained on the combined dataset outperformed the 2D descriptor models, both achieving a test RMSE of 0.92.

<br></br>
<img src=figures/test_mix_models.png>

**Figure 8: RMSE values for models trained on different feature sets and tested on the test set.**
<br></br>

### Error analysis

The parity plot shows that most predictions are close to the identity line and that incorrect predictions mostly occur for solubility values far away from the dataset mode (**Fig. 9**). The 10 most overestimated and 10 most underestimated cases in the test set were plotted (**Fig. 10**). Note that there were some instances that actually consisted of multiple compounds and were named like "methane; sulfuric acid" (2nd row, 2nd column). Filtering out these multiple compounds from the data set could improve the ability of the model to generalize solubility for organic compounds.

<br></br>
<img src=figures/parity_plot_2.png>

**Figure 9: Parity plot for the voting model using both 2D descriptors and MACCS on the test set.**
<br></br>

<br></br>
<img src=figures/error_structures.png>

**Figure 10: Top 10 under- and overestimated compounds with corresponding solubility values and predicted solubility values.**
<br></br>

## Graph Neural Networks

Graph neural networks were used to predict the solubility of compounds. Previous models struggled with instances containing multiple chemical compounds, which also resulted in unconnected graphs. These problematic instances were removed from the data set.

The graphs were constructed from SMILES strings, which were transformed into RDKit molecular objects to obtain adjacency matrices. Node features included one-hot encodings for element types and hybridization states, as well as vectors describing atomic properties such as degree, formal charge, implicit valence, mass, and aromaticity. Edge features were encoded to indicate bond types (single, double, triple, aromatic) and properties (part of a conjugated system or ring). This setup provided the basis for testing different graph neural network architectures.

To explore different architectures, graphs with node and edge features only were used to train a graph convolutional network (GCN), a graph attention network (GAN) with convolutional layers incorporating attention mechanisms, and a transformer.

Since the previous section demonstrated the power of 2D descriptors as features, the same set of 2D descriptors (MolWt, MolLogP, MolMR, HeavyAtomCount, etc.) were used as graph-level features.The same GCN architecture was used as before, but after the graph convolutional layer, the graph-level features were incorporated into a dense network.

Table 2 indicates that incorporating graph-level features into the graph convolutional network significantly improves its performance, achieving the lowest RMSE value of 0.921 compared to other models on the test set.

**Table 2: RMSE values for graph neural networks on test set.**
| Model                                      | RMSE  |
|--------------------------------------------|-------|
| Graph Convolution                          | 1.110 |
| Graph Attention                            | 1.066 |
| Transformer                                | 1.096 |
| Graph Convolution (+ graph-level features) | 0.921 |

Taken together, the graph neural networks did not perform better than the models using MACCS fingerprints and 2D descriptors.

## Literature

Sorkun, M.C., Khetan, A. and Er, S. (2019) ‘AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds’, Scientific Data, 6(1), p. 143. Available at: https://doi.org/10.1038/s41597-019-0151-1.
