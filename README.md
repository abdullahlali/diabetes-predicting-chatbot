# **SugarSage: A Diabetes Prediction and Information Chatbot**

**SugarSage** is a Python-based interactive chatbot designed to predict the probability of diabetes and provide comprehensive answers to diabetes-related questions. The prediction model uses a multiple regression algorithm trained on a diabetes dataset, while the chatbot component answers user queries using NLP techniques.

---

## **Project Overview**

This project has two main components:
1. **Diabetes Prediction Model**: Uses multiple regression to predict the probability of diabetes based on user inputs like age, BMI, blood glucose level, etc.
2. **Interactive Q&A Chatbot**: Provides information about diabetes symptoms, management, treatment, and prevention.

The dataset used to build the model is sourced from **Kaggle** (provided in the repository). Multiple regression was performed using **Excel** with the **Analysis ToolPak** add-in to identify coefficients for the model.

---

## **How the Diabetes Prediction Model Was Built**

1. **Dataset**:
   - The dataset used for training the model is sourced from [Kaggle.com](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and is included in the repository.
   - It contains features such as:
     - Gender
     - Age
     - Hypertension
     - Heart Disease
     - Smoking History
     - BMI (Body Mass Index)
     - HbA1c Level
     - Blood Glucose Level

2. **Multiple Regression with Excel**:
   - The data was loaded into **Excel**.
   - Using the **Analysis ToolPak**, a multiple regression analysis was performed to determine the relationship between the independent variables (features) and the dependent variable (**diabetes** probability).
   - The resulting coefficients from the regression model were extracted and used in the Python program to make predictions.

3. **Model Equation**:
   The multiple regression equation for predicting diabetes is as follows:
   - **Y** is the output value (diabetes probability).
   - The coefficients are directly derived from the multiple regression analysis in Excel.

4. **Implementation**:
   - The equation is implemented in Python to calculate the probability of diabetes based on user inputs.
   - The output is passed through a **sigmoid function** to ensure the probability lies between 0 and 1.
   - The model achieved an R-squared value of **0.31**, which is typical for data involving biological variables.

---

## **Key Features**

1. **Diabetes Prediction**
   - Predicts the likelihood of diabetes based on user-provided health data.
   - Offers actionable advice if the probability exceeds a certain threshold.

2. **Interactive Q&A Chatbot**
   - Provides answers to a wide range of diabetes-related questions, such as symptoms, complications, treatments, and prevention strategies.
   - Utilizes natural language processing (NLP) techniques with **TF-IDF vectorization** and an **SVM model** to match user questions to predefined answers.
   - The chatbot achieved an accuracy of **96%**, ensuring reliable responses to user queries.

3. **User-Friendly Interface**
   - Simple inputs for diabetes prediction.
   - Conversational chatbot interface to ask follow-up questions.

4. **Sentiment Analysis for Feedback**
   - Collects user feedback at the end of the session.
   - Analyzes sentiment using the **TextBlob** library to improve the user experience.

---

## **Technologies Used**
- **Python**
- **Excel Analysis ToolPak** (for multiple regression analysis)
- **Natural Language Toolkit (nltk)**
- **Support Vector Machine (SVM)**
- **TF-IDF Vectorization**
- **Multiple Regression**
- **TextBlob for Sentiment Analysis**

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SugarSage.git
   cd SugarSage
   pip install numpy sklearn nltk textblob
   python3 sugarSage.py
   ```

