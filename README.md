
# Project Overview
This repository houses the code behind the approach presented in the paper [*'Exploration of Algorithmic Trading Strategies for the Bitcoin Market'*](https://arxiv.org/abs/2110.14936).
This paper describes research conducted in using machine learning algorithms to predict the price of Bitcoin on a day-to-day basis.

The key contributions of this paper can be summarised as follows:
* We attempt to verify the performance of models reported in previous studies and demonstrate issues with respect to overfitting, something which is very common in published data science research.
* To the best of our knowledge, we establish a plausible new benchmark for machine learning in this space by representing the most rigorous and credible scientifically published Bitcoin trading algorithm at the time of publishing.
* We develop an algorithmic Bitcoin trading method using a broad range of features which, when taken together, have been underexploited in previous studies.
  The use of these features in their raw form does not prove to have a significant effect on the model performance.
* We demonstrate the real-world trading performance of our developed model through an empirical test on completely unseen data collected during the year 2021.
  This provides a reliable metric for evaluating model performance over and above conventional metrics.
* We utilise the probabilistic outputs of the classifiers to naturally parameterise trading risk and allowing traders to flexibly specify their risk appetite.
  This shows traders with higher risk tolerances to be the most profitable.

This research was originally undertaken as my final year project in university, however, after receiving a first class honours degree for this work, I subsequently decided to publish it as a scientific paper.


# Repository Breakdown
This repository is broken down into two sections: [*baseline_paper_implementation*](/baseline_paper_implementation) and [*my_approach*](/my_approach).

### 1. [*Baseline Implementation*](/baseline_paper_implementation)

The [baseline_paper_implementation](/baseline_paper_implementation) section contains the code behind the initial work undertaken as part of this project, whereby I set out to reproduce the results from previous works in this area to use it as a baseline for my approach.
One prominent paper in the industry titled [*'Time-series forecasting of Bitcoin prices using high-dimensional features: a machine learning approach'*](https://link.springer.com/article/10.1007/s00521-020-05129-6) claimed to achieve significant results in their prediction efforts and their code was published on GitHub [here](https://github.com/heliphix/btc_data), so I decided to use this to form a baseline for my research.

After cloning this repository, my process involved understanding the code, analysing the approach, formatting and refactoring the code, fixing bugs, running the models to get results, and expanding the codebase to test other models and their performance.

The models used in the previous paper were:
 - Artificial Neural Network (ANN),
 - Stacked Artificial Neural Network (SANN),
 - Long Short-Term Memory Network (LSTM),
 - Support Vector Machine (SVM),

But the models tested in this folder were expanded to also include:
 - Stochastic Gradient Descent Classifier (SGDC),
 - eXtreme Gradient Boosting (XGB) model.

Alongside this, the evaluation metric we wanted to use as part of this research was one that would expose overfitting and reflect the real-world viability of a model in a trading environment, i.e. when faced with completely unseen data.
As such, accompanying these models and the refactored code in this folder is a notebook to conduct a real-world test of the models on real-world Bitcoin data not featured in the training or test set.
This notebook sets out to use each model to trade bitcoin and to evaluate them using profit and loss.

Upon analysing the results we were able to recreate using the models used in the previous paper, we noticed the results were not as good as those presented in the paper and that a rethinking of data and approach may be needed.
Nevertheless, the work done in the previous paper acted as a useful measure to improve upon.

### 2. [*My Approach*](/my_approach)
The [my_approach](/my_approach) section contains the code representing the approach proposed in the underlying paper for this piece of work.
This approach sets out to improve on the results presented in the baseline by utilising additional data collection steps, different data processing approaches, and new modelling techniques.

We saw in the baselines that Neural Network models were quite ineffective.
We speculate that this is due to the limitation in data we had available, with only 10yrs * 365days worth of training instances available.
As such, we decided to focus on tuning simpler models to outperform the baseline models.
The models run in this section include:
 - Support Vector Machine (SVM),
 - eXtreme Gradient Boosting (XGB) model,
 - Random Forest Classifier (RFC).

The same real-world trading test involving completely unseen data is used on the models as a means of evaluation. 


# Folder Breakdown
Each of these two folders follow a very similar structure.
The following shows the directory architecture of the [my_approach](/my_approach) folder alongside a brief explanation of each sub-folder:
```
├───Code
│   ├───Classification_models        - Code to train classification models on the data and to evaluate these models
│   ├───Collection_and_processing    - Code to scrape the dataand process it
│   └───Feat_select                  - Code to perform feature selection on the processed data
│
├───Config_files
│
├───Data
│   ├───Feat_select                  - CSV's containing the data after feature selection
│   ├───Max_conf_vals                - CSV's containing each model's maximum and minimum predictive confidence scores
│   ├───Model_evaluation             - CSV's containing each models:
│   │   ├───Cross_val_scores             * cross validation accuracy and F1 scores
│   │   └───Training_predictions         * predictive performance on the training data 
│   ├───Processed_data               - CSV's containing the data post-processing
│   └───Scraped_data                 - CSV's containing the raw scraped data
│
└───Trained_models
    ├───Best_param_files             - Json's containing each model's hyperparameter optimisation parameter results
    ├───Classification               - Heirarchichal data formats (HDF5) containing the trained classification models
    ├───PCA                          - Joblib files containing the trained PCA models
    └───Scales                       - Joblib files containing the pipelines used to scale the data
```

The [baseline_paper_implementation](/baseline_paper_implementation) roughly follows this same structure, however, it also contains some code for carrying out exact predictions of the next day bitcoin price by means of regression analysis.


# Setup
When attempting to run the code in this repo, there are a number of things you must do to ensure you are able to run this from start to finish successfully.

1. **Download chromedriver**
   * This is needed to carry out one of the internal Bitcoin data scrapes in the data collection step.
   * You can download this from https://chromedriver.chromium.org/downloads.
   * The version of chromedriver that you download must match the version of Google Chrome that you have on your machine.
     * Find version: open chrome >> three dots in top right >> settings >> About Chrome.
   * Put this downloaded chromedriver in `C:\Program Files (x86)\Google\Chrome\Application`.
     - Chromedriver can be put anywhere, but the `chromedriver_location` paths in the `scrape_config` YAML configuration file must be updated.


2. **Set up an account with Quandl and set API key as an environment variable**
   * This is needed to carry out one of the economic data scrapes in the data collection step.
   * An account can be set up at https://data.nasdaq.com/sign-up.
   * Set your quandl api key as an environment variable on your machine under the name `QUANDL_API_KEY`.
     * Find your API key: log into Quandl account >> click profile icon in top right >> account settings.
   * The environment variable process ensures no API key is hardcoded into the code.


3. **Set up a Twitter developer account and set API keys as environment variables**
   * Set up a Twitter account at https://twitter.com/.
   * Log in to your Twitter account and go to the [twitter developer portal](https://developer.twitter.com/en/portal/projects/1349751881988530180/apps/19866167/keys).
   * Set up a project and take note of the API keys and tokens that are generated.
   * Set the generated API keys and access tokens as environment variables on your machine under their respective names:
     - `TWITTER_API_ACCESS_TOKEN`
     - `TWITTER_API_EARER_TOKEN`
     - `TWITTER_API_KEY`
     - `TWITTER_API_SECRET_ACCESS_TOKEN`
     - `TWITTER_API_SECRET_KEY`
   * The environment variable process ensures no API key is hardcoded into the code.


4. **Check for more Elon Musk tweets**
   * When scraping Twitter data, I had difficulty with the API limits and resorted to downloading a zip file of Elon Musks tweets from https://www.kaggle.com/ayhmrba/elon-musk-tweets-2010-2021.
   * Check to see if the zip file at this URL has been updated to contain tweet data after 23-03-2021.
   * If it has updated, update the `musk_tweets_till_2021_data.csv` file in the [Data/Scraped_data](/my_approach/Data/Scraped_data) folder to the updated version.


5. **Adjust the file paths in the YAML configuration files**
   * The `downloads_folder` filepath will need to be changed in the _Config_files/scrape_config.yaml_ file to adapt this path to access your Downloads folder.
   * The `data_directory`, `model_directory`, and `audio_file_path` filepaths also need to be changed in the _Config_files/config.yaml_ file to adapt these paths to access your Documents folder.
   * This change will need to be applied in the [baseline_paper_implementation](/baseline_paper_implementation) folder and the [my_approach](/my_approach) folder.


# Modelling Workflow
The workflow to running the code in this repo can be defined in 5 steps.
Over the course of carrying out these steps, the files in the `Data` and `Trained_models` directories will be populated.

1. Scrape the data
   * Run all code in _Code/Collection_and_processing/Data_collection.ipynb_
2. Process the data
   * Run all code in _Code/Collection_and_processing/Data_processing.ipynb_
3. Carry out feature selection
   * Run all code in _Code/Feat_select/PCA.ipynb_
4. Train the classification models
   1. Scale the data
      * Run _sections 1,2,3_ in _Code/Classification_models/Classification_model_training.ipynb_
   2. Define optimal parameters using hyperparameter optimisation
      * Run _section 4_ in _Code/Classification_models/Classification_model_training.ipynb_
   3. Check how AutoML defines the best models and parameters
      * Run _section 5_ in _Code/Classification_models/Classification_model_training.ipynb_
   4. Train models using these parameters
      * Run _section 6_ in _Code/Classification_models/Classification_model_training.ipynb_ part
5. Test the classification models when trading in the real-world
   * Run _Code/Classification_models/Real_world_eval.ipynb_
