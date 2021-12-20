# ABSTRACT
CerraNet is a deep learning convolutional neural network, specially designed to process and contextually classify the types of use and coverage in the Brazilian Cerrado, whose biome is the second largest in Brazil and is characterized by different landscapes, sometimes forests, sometimes savannas and countryside. However, only four classes were considered, Forest, Deforest, Fire and Agriculture. The model was trained with images obtained by the CBERS-4A satellite, with two meters of spa- tial resolution, totalling 32000 images; for tests, two other datasets were structured, each with 800 images, differentiating only the observed areas and the spatial resolution, the first with two meters and the second with eight meters. Regarding the model’s architecture, a model was designed with six convolutional layers, succeeded by average pooling and dropout layers, as well as two dense and dropout layers, optimized with Adam. Model performance was evaluated using Accuracy and F1-Score metrics, achieving respectively for each test set, 94.38%, 94.37%, 46.75% and 46.75% accuracy in image classification. In general, the model achieved encouraging results, above all, it proved to be optimized and efficient.

# GENERAL INFORMATIONS
