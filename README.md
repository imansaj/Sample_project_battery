# Real-life-based estimation of the State Of Health of battery cells by machine learning 


Table of Contents
=================

[Running notes](#running-notes)

[Data information](#data-information)

[Goal](#Goal)

[Problem definition](#problem-definition)

[Methods](#methods)

[Sections](#sections)

[Feature selection and cross validation](#feature-selection-and-cross-validation)

[Model](#model)

[Results](#results)

[Post analysis](#post-analysis)


### Running notes
I set the main folder for saving the data as 'D:/Severson_battery_data/' in the args.main_path.
Please kindly change it as you desire. 

The code will try to download the data which is around 3.01G.

The first run will take longer(around 5 min on my PC) due to downloading and resampling. The next runs will be much faster(around 87 seconds). 

### Data information
The data information can be found here:
https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

### Goal
Find the capacity drop of the battery using machine learning since it cannot be measured in the vehicle. 

### Problem definition

The battery cells experience a loss of capacity as they are used. This is shown in Figure 1, as time passes the
unusable part, which is shown as rock content gets bigger, so we will have a lower capacity. This effect can be seen
as fast charge and discharge of the battery. In other words, the active part is filled and drained much faster compared to
the new cell. These phenomena are called aging, and the term that we are usually more familiar with is battery health 
(for example in the iPhones.) 

![figure 1](https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Aging%20schematic.png)

Figure 1. The schematic of a cell after partial aging.

Now the problem here is that this active part that we call "the capacity" cannot be measured easily, and it needs lab
equipment. Basically, this means that you cannot have the current capacity or battery health by direct measurement, and 
we need a model to predict it using the features that can be measured directly like voltage. So, we define our goal as 
below.

```
Goal: Find a model that can predict the capacity of the cell using only its voltage.
```

It should be noted that using voltage as the input is based on the pack design. A pack is defined as a 
pack of cells that are assembled in series or parallel to reach a desired output. If we assemble the cells in 
parallel we will only have their voltage and if we assemble them in series, we will only have their current. I am using 
Voltage since it was like the pack design that we had. 

### Methods

First, to get a better understanding of the problem, we plot the voltage(our input) (Figure 2) for one of the cells as the
cell ages and the capacity (the output) (figure 3) for all the cells.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_full.jpg" width="600">

Figure 2. The evolution of one voltage cycle at different aging times for cell b1c0.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Capacity_vs_cycles.png" width="600">

Figure 3. The capacity versus cycle number for different cells.

As can be seen from Figure 2 we can see that although there is a trend that the curves are getting smaller for a larger 
number of cycles, this alone is not sufficient to predict the aging, and we need more reasonable features. To get a 
better physical understanding of the problem we plot one cycle of voltage and current, and we explain how we can extract
more useful information from these curves.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/one_cycle.jpg" width="600">

Figure 4. The defined sections of one cycle.

As can be seen from Figure 4 we can divide a voltage curve into 5 regions each with a physical meaning. The first three 
regions are associated with charging. The fourth region is the discharging section. and the fifth region is the rest section.
We start with the charging section. The common charging method for battery cells is constant-current constant-voltage
method. In this method the cell is charged by a constant current(or multi-constant current with different values) until 
it reaches a certain voltage. After that we keep charging the cell while keeping its voltage constant until the current 
reaches zero. As can be seen from figure 4 the constant current part which we named sections I and II is made 
from three constant currents. For simplicity, we merged two of them in section I. Section III is the constant voltage 
region. Section IV is the discharge region. And section V is the rest region when the current is zero and the cell is at
its minimum voltage. In summary, we have these definitions for sections:

### Sections

Section I: Charging region, Constant-current part 1

Section II: Charging region, Constant-current part 2

Section III: Charging region, Constant-voltage

Section IV: Discharging region

Section V: Rest region

Now we plot each of these sections for different cycles to see how they change as the cell ages. We also plot their 
slopes to see if these sections change faster as the cell ages.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_Section%20I.jpg" width="600">

Figure 5. The voltage of section I versus time at different cycle numbers.


<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_Section%20I%20slopes.jpg" width="600">

Figure 6. The slope of the voltage of section I versus time at different cycle numbers.

The plots for other sections can be found in the "plots/sections/" folder in the "args.main_path" defined by the user. 
We can see from Figures 5 and 6 that there is indeed a gradual change in these regions as the cell ages. We have one 
capacity value for each cycle. But in each region, we have a curve for each cycle. To reduce the dimensionality, we apply
7 functions on each curve (These are known as summary statistics):
- Sum
- Duration
- Standard deviation (std)
- Lower bound (LB): (mean – std)
- Upper bound (UB): (mean + std)
- Kurtosis
- Skew

After applying these functions, we will have a better understanding of how each region is changing compared to capacity.
These will be our final features. We have 5 regions and for each 7 features for slope and non-slope which makes it a total 
of 5x7x2 = 70 features. In Figure 7 I plotted the features for Region 1. As can be seen, there is a relation between the
capacity and extracted features.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/features_serial(b1c0)_Voltage(V)__I.jpg" width="600">

Figure 7. The features for Region 1. The number in the parentheses shows the mutual information between capacity and the
corresponding feature.

### Feature selection and cross-validation
Using all of these features leads to poor model performance due to the high number of features known as the curse of 
dimensionality. To avoid this, we reduce the number of features by using an algorithm called mutual information. 

Mutual information calculates the dependency of two variables. The mutual information for two variables “a” and “b”, 
gives us the information that we can get about “b” by investigating “a” and vice versa. By using this method, we can 
find the most productive features in a set of features and only use them for training the model. So, with this method, 
each feature (variable a) is compared to the capacity (variable b), and the features with the highest mutual information
with capacity are used. More details are provided in the “Model and Results” section. I used top 37 features selected 
for this algorithm. The number 37 is found by trial and error.

Since the number of selected cells is small, to ensure that the model is generalizable we used cross-validation to 
validate the model. For this method, the data were divided into 5 equal parts and the model was trained and tested for
each part separately. This means that each cell is treated as an unseen cell for each iteration of the cv algorithm, and 
therefore our model performs much better in real practice. 

### Model
Since most of the hard work regarding the data science part is done on the feature engineering, most of the regression
models in the scikit-learn will give good results. But for two reasons I used lightGbm: 

- I needed a tree-based model, so I could find the feature importance (using shap) for the post-analysis of the results.
- Among the tree-based models, lightGBM is the lightest both in terms of speed and the amount of space it uses, which 
makes it ideal for real-life applications to be loaded on chipsets.

### Results
In Figures 8 and 9 I plotted the results. For plotting these curves, the results of cross-validation are saved at each iteration
and then all the curves are plotted together. 

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/final_results_list(0).jpg" width="600">

Figure 8. The predicted capacity of the first group of 6 cells using the proposed model. For plotting I split the 12 cells into 
two groups of 6 for better visualization of the results. The cells were randomly rotated into two groups for training and
testing. All the curves plotted here are from the test dataset.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/final_results_list(1).jpg" width="600">

Figure 9. The predicted capacity of the second group of 6 cells using the proposed model. For plotting I split the 12 cells 
into two groups of 6 for better visualization of the results. The cells were randomly rotated into two groups for 
training and testing. All the curves plotted here are from the test dataset.

The mean of RMSE on cross-validated parts is 0.00395 with a standard deviation of 
0.00116. The mean of the r-squared score was one with a standard deviation of 0.0.

### Post analysis
For the post-analysis of the results, we can plot the feature importance which gives us an idea of how important 
each feature is. The idea behind feature importance is that, once we have a working model, we can give the top selected features along with the training data to shap algorithm, and it gives us the importance of each feature. Then we can group features based on a desired facto and sum their importance to see the importance of each factor. For example, in Figure 10 I plotted the importance of each section in the final results. As can be seen, sections II and I have the highest importance. This means that the charging part of the curve has the highest importance score for predicting the capacity. Therefore in practice, we can put extra measures for measuring the charging part more accurately for better prediction of capacity.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/features_importance_region_Section.png" width="600">

Figure 10. The Feature importance of each section is calculated by shap.


