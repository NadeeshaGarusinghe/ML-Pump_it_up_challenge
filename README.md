# ML-Pump_it_up_challenge
The "Pump it Up: Data Mining the Water Table" (competition link: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

This problem basically addresses the problem of lacks access to clean water for the 25 million of population in Tanzania. There are 4 different data sets; submission format, training set, test set and train labels set which contains status of wells. 

# Deal with missing values
There were 7 features that contain null,zero and missing values.
1. funder
2. installer
3. subvillege
4. public_meating
5. scheme_management
6. scheme_name
7. permit
Mainly, null, zero and missing values changed to mean or collected in a group as named unknown according to nature of the column.

# Drop similar features
(extraction_type, extraction_type_group, extraction_type_class), 
(payment, payment_type), 
(water_quality, quality_group), 
(source, source_class), 
(subvillage, region, region_code, district_code, lga, ward), 
(waterpoint_type, waterpoint_type_group) 
(scheme_name, scheme_management)

This set of features contained very similar information, so the correlation between them is high. It could be a reason to overfitting the training data by including all the features in the model.
So, I dropped the 'extraction_type_group','extraction_type_class','payment','water_quality', 'source_class','subvillage','region','lga','ward', 'installer', 'wpt_name', 'scheme_name', 'waterpoint_type_group',

And also, as shown in the heatmap below, there exists quite a strong correlation between district_code and region_code, so we will drop one of these variables. The negative correlation to the target variable of the "region_code" is higher than that of the "district_code". So, I Kept the variable with higher correlation to the target.
![Capture](https://user-images.githubusercontent.com/47114134/132943642-7eff5825-57aa-42a6-a938-873a5a0a9d1b.JPG)

# Reduce cardinality
The columns in which values can be ordered I performed an Ordinal encoding:
1. quality_group
2. quantity_group
3. payment_type

The cardinality of the following 2 features are reduced to 10 and then one-hot encodeed them.
1. scheme_managenemt
2. extraction_type

The cardinality is very high of the following 3 features.
1. funder
2. installer
3. subvillage
I first drop those features and train a model. later, tried some other ways like, frequency encoding, binary encoding etc.

Other features were one-hot encoded as the cardinality is lower than 10:
1. public_meeting 
2. permit 
3. source_class
4. management_group
5. waterpoint_type_group
6. source_type
7. basin

# Feature engineering
The 'year' feature is converted to a feature called 'decades' for future encoding. 
Zero shows the missing values. This have majority of the data set so, it will not be changed to the mean or median, kept as new value in decades.
   
# Ordinal encoding of categorical data
Several categorical features can be encoded in a specific order that follows from the range of its values. By using ordinal encoding instead of one-hot encoding I avoided creating numerious additional columns and provide some logic to the model on how to evaluate these features. 
  Ex: the quality_group variable-> the higher the label, the better the water quality, the more likely a pump is functional.
I did ordinal encoding for  the following features.
1. Quality_group
2. Quantity_group
3. Payment_type
4. Public_meeting
5. Permit

# Model selection
I tired several ML models for classification. Logistic Regression, RandomForest Clasifier, CatBoost Classifier and XGBoost Classifier are some of them.
The best model was obtained by XGBoost Classifier with SMOTE over-sampling technique.
