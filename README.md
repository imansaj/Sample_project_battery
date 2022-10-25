# Estimation of State Of Health of battery cells by machine learning using experimental data.

### Running notes
I set the main folder for saving the data as 'D:/Severson_battery_data/' in the args.main_path.
Please kindly change it as you desire.

The code will try to download the data which is around 3.01G.

### Data information
The data information can be found here:
https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

### Problem definition

The battery cells experience loss of capacity as they are used. This is shown in figure 1. as the time passes the
unusable part, which is shown as rock content gets bigger, and so we will have a lower capacity. This effect can be seen
as fast charge and discharge of the battery. In other words, the active part is filed and drained much faster compared to
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

It should be noted that using voltage as the input is based on the pack design. Pack is defined as a 
pack of cells that are assembled in series or parallel to reach a desired output. If we assemble the cells in 
parallel we will only have their voltage and if we assemble them in series, we will only have their current. I am using 
Voltage since it was like the pack design that we had. 

### Methods

First to get a better understanding of the problem, we plot the voltage(our input) (figure 2) for one the cells as the
cell ages and the capacity (the output) (figure 3) for all the cells.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_full.jpg" width="600">

Figure 2. The evolution of one voltage cycle at different aging times for cell b1c0.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Capacity_vs_cycles.png" width="600">

Figure 3. The capacity versus cycle number for different cells.

As can be seen from figure 2 we can see that although there is a trend that the curves are getting smaller for larger 
number of cycles, but this alone is not sufficient to predict the aging, and we need more reasonable features. To get a 
better physical understanding of the problem we plot one cycle of voltage and current, and we explain how we can extract
more useful information from these curves.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/one_cycle.jpg" width="600">

Figure 4. The defined sections of one cycle.

As can be seen from figure 4 we can divide a voltage curve into 5 regions each with a physical meaning. The first three 
regions are associated with charging. The fourth region is discharging section. and the fifth region is the rest section.
We start with the charging section. The common charging method for battery cells is constant-current constant-voltage
method. In this method the cell is charged by a constant current(or multi constant current with different values) until 
it reaches a certain voltage. After that we keep charging the cell while keeping its voltage constant until the current 
reaches zero. As can be seen from figure 4 the constant current part which we named sections I and II is made 
from three constant currents. For simplicity, we merged two of them in section I. Section III is the constant voltage 
region. Section IV is the discharge region. And section V is the rest region when the current is zero and the cell is at
its minimum voltage. In summary, we have these definitions for sections:

##### Sections

Section I: Charging region, Constant-current part 1

Section II: Charging region, Constant-current part 2

Section III: Charging region, Constant-voltage

Section IV: Discharging region

Section V: Rest region

Now we plot each of these sections for different cycles to see how to they change as the cell ages. We also plot their 
slopes to see if these sections change faster also as the cell ages.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_Section%20I.jpg" width="600">

Figure 5. The voltage of section I versus time at different cycle numbers.


<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/Sections_serial(b1c0)_Voltage(V)_Section%20I%20slopes.jpg" width="600">

Figure 6. The slope of the voltage of section I versus time at different cycle numbers.

The plots for other sections can be found in the "plots/sections/" folder in the "args.main_path" defined by the user. 
We can see from figures 5 and 6 that there is indeed a gradual change in these regions as the cell ages. We have one 
capacity value for each cycle. But in each region, we have a curve for each cycle. to reduce the dimensionality, we apply
7 functions on each curve (These are known as summary statistics):
- Sum
- Duration
- Standard deviation (std)
- Lower bound (LB): (mean – std)
- Upper bound (UB): (mean + std)
- Kurtosis
- Skew

After applying these functions, we will have a better understanding of how each region is changing compared to capacity.
These will be our final features. we have 5 region and for each 7 features for slope and non-slope which makes it total 
of 5x7x2 = 70 features. In figure 7 I plotted the features for region 1. As can be seen there is a relation between the
capacity and extracted features.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/features_serial(b1c0)_Voltage(V)__I.jpg" width="600">

Figure 7. The features for region 1. The number in the parentheses shows the mutual information between capacity and the
corresponding feature.

##### Feature selection and cross validation
In general, it is not a good idea to use all the feature, since it will decrease the performance of the model. Here I 
used mutual information to select the most important features. I used top 37 features selected for this algorithm. The 
number 37 is found by trial and error.

Since the number of selected cells are small, to ensure that the model is generalizable we used cross-validation to 
validate the model. For this method, the data were divided into equal parts and the model was trained and tested for
each part separately. The mean of accuracy of cross-validation parts shows how our method performs on unseen data. I 
used 5-fold splits for 12 cells.

### Results
In figure 8 I plotted the results. For plotting these curves, the results of cross validation are saved at each iteration
and then plotted together. The other plots can be found in the plots' folder.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/final_results_list(0).jpg" width="600">

Figure 8. The predicted capacity of each cell using the proposed model. The cells were randomly rotated into two groups 
for training and testing. All the curves plotted here are from the testing group.

I used lightGBM for the modelling. The mean of RMSE on cross-validated parts is 0.00395 with a standard deviation of 
0.00116. We can also plot the feature importance which gives us an idea of how important each feature is. For example,
in figure 9 I plotted the importance of each section in the final results.

<img src="https://github.com/imansaj/Sample_project_battery/blob/main/Documentation%20material/features_importance_region_Section.png" width="600">

Figure 9. The Feature importance of each section calculated by shap.

### Conclusion
In summary, here I showed that with research-based feature engineering we can solve a difficult problem with low loss
and high performance. 

