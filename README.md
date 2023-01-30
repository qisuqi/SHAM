# SHAM

In the SHAM model, each feature is weighted based on its
previously calculated SHAP values to improve the NAM
architecture by leveraging domain knowledge given by the
SHAP method. As a result of this simple alteration to the
NAM architecture, training can be restricted by a previous
explanation of the dataset as domain knowledge. The integration of explanations as domain knowledge leads NAM,
an interpretable model, to improve on performance metrics.

An example of using SHAM is in SHAM_EvoSynth.ipynb

<img width="359" alt="SHAM_architecture" src="https://user-images.githubusercontent.com/63663984/215523722-c760ba36-61a7-4094-954e-5903bfc03b32.png">
