import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

# Download all requirements only once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Extended Diabetes Q&A Dataset
dataset = [
    # Basics and Types
    ("What is diabetes?", "Diabetes is a chronic condition that occurs when the body cannot properly process blood sugar (glucose) due to lack of insulin or insulin resistance."),
    ("What are the main types of diabetes?", "The main types of diabetes are Type 1 diabetes, Type 2 diabetes, and gestational diabetes."),
    ("What is Type 1 diabetes?", "Type 1 diabetes is an autoimmune condition where the body's immune system attacks insulin-producing beta cells in the pancreas."),
    ("What is Type 2 diabetes?", "Type 2 diabetes occurs when the body becomes resistant to insulin or does not produce enough insulin to manage blood sugar levels."),
    ("What is gestational diabetes?", "Gestational diabetes is a type of diabetes that develops during pregnancy and usually disappears after giving birth."),
    ("What is maturity-onset diabetes of the young (MODY)?", "MODY is a rare form of diabetes caused by a mutation in a single gene, often diagnosed in adolescence or early adulthood."),
    ("What is latent autoimmune diabetes in adults (LADA)?", "LADA is a slow-progressing form of Type 1 diabetes that occurs in adults, often misdiagnosed as Type 2 diabetes."),
    ("What is secondary diabetes?", "Secondary diabetes is caused by another medical condition or medication, such as pancreatitis or steroid use."),
    ("What is prediabetes?", "Prediabetes is a condition where blood sugar levels are higher than normal but not high enough to be classified as diabetes."),
    ("What is insulin resistance?", "Insulin resistance occurs when the body's cells stop responding effectively to insulin, leading to high blood sugar levels."),
    
    # Symptoms and Diagnosis
    ("What are the early symptoms of diabetes?", "Early symptoms include increased thirst, frequent urination, fatigue, unexplained weight loss, and blurry vision."),
    ("What are the warning signs of diabetic ketoacidosis (DKA)?", "Signs include nausea, vomiting, fruity-smelling breath, rapid breathing, and confusion."),
    ("How is diabetes diagnosed?", "Diabetes is diagnosed using tests like fasting blood glucose, oral glucose tolerance, and HbA1c tests."),
    ("What is a random blood glucose test?", "A random blood glucose test checks blood sugar levels at any time of the day, regardless of when you last ate."),
    ("What is an oral glucose tolerance test (OGTT)?", "The OGTT checks blood sugar levels after fasting and again two hours after consuming a glucose-rich drink."),
    ("What are normal blood sugar levels?", "Normal fasting blood sugar levels are 70-100 mg/dL, and post-meal levels should be below 140 mg/dL."),
    ("What is the role of HbA1c in diabetes diagnosis?", "The HbA1c test measures average blood sugar over 2-3 months. A result of 6.5% or higher indicates diabetes."),
    
    # Treatment and Management
    ("How is Type 1 diabetes treated?", "Type 1 diabetes is treated with insulin therapy, blood sugar monitoring, a healthy diet, and regular exercise."),
    ("How is Type 2 diabetes treated?", "Type 2 diabetes is managed with lifestyle changes, medications like metformin, and sometimes insulin therapy."),
    ("What are the different types of insulin?", "Types of insulin include rapid-acting, short-acting, intermediate-acting, and long-acting insulin."),
    ("What is an insulin pump?", "An insulin pump is a small device that delivers a continuous supply of insulin into the body."),
    ("What is continuous glucose monitoring (CGM)?", "CGM is a technology that tracks blood sugar levels in real-time using a sensor placed under the skin."),
    ("What is an artificial pancreas?", "An artificial pancreas is a device that automatically monitors and regulates blood sugar using an insulin pump and CGM."),
    ("What is the role of exercise in diabetes management?", "Exercise improves insulin sensitivity, lowers blood sugar, and helps maintain a healthy weight."),
    ("What is medical nutrition therapy (MNT)?", "MNT is a personalized dietary plan created to help manage diabetes and control blood sugar levels."),
    ("How do oral diabetes medications work?", "Oral medications like metformin reduce glucose production in the liver or help the body use insulin more effectively."),
    ("What is the impact of weight loss on Type 2 diabetes?", "Losing even 5-10% of body weight can improve insulin sensitivity and lower blood sugar levels."),
    
    # Complications
    ("What are the long-term complications of diabetes?", "Complications include heart disease, kidney damage, nerve damage, eye problems, and poor wound healing."),
    ("What is diabetic neuropathy?", "Diabetic neuropathy is nerve damage caused by high blood sugar, leading to tingling, pain, or numbness in the extremities."),
    ("What is diabetic retinopathy?", "Diabetic retinopathy is damage to the blood vessels in the eyes, which can lead to vision loss."),
    ("What is diabetic nephropathy?", "Diabetic nephropathy is kidney damage caused by poorly controlled diabetes over time."),
    ("What is diabetic foot syndrome?", "Diabetic foot syndrome involves ulcers, infections, and poor wound healing due to nerve and blood vessel damage."),
    ("What is hyperosmolar hyperglycemic state (HHS)?", "HHS is a life-threatening complication of Type 2 diabetes involving very high blood sugar and dehydration."),
    ("How does diabetes affect the immune system?", "High blood sugar can weaken the immune system, making diabetics more prone to infections."),
    
    # Prevention
    ("How can Type 2 diabetes be prevented?", "Prevention strategies include maintaining a healthy weight, eating a balanced diet, staying active, and avoiding smoking."),
    ("What is the role of fiber in diabetes prevention?", "High-fiber foods can slow glucose absorption, reducing blood sugar spikes and improving insulin sensitivity."),
    ("Can intermittent fasting help prevent diabetes?", "Intermittent fasting may improve insulin sensitivity and help with weight loss, reducing diabetes risk."),
    
    # Lifestyle and Mental Health
    ("How does stress impact diabetes?", "Stress can raise blood sugar levels due to the release of hormones like cortisol."),
    ("Can sleep affect blood sugar levels?", "Yes, poor sleep quality can worsen insulin resistance and lead to higher blood sugar levels."),
    ("How does alcohol affect diabetes?", "Alcohol can cause blood sugar fluctuations, including hypoglycemia if consumed on an empty stomach."),
    ("How can I quit smoking to improve diabetes management?", "Quitting smoking improves circulation, insulin sensitivity, and reduces complications like heart disease."),
    ("How does diabetes impact mental health?", "Diabetes can increase the risk of depression, anxiety, and diabetes distress due to the burden of disease management."),
    
    # Advanced Topics and Research
    ("What is diabetes remission?", "Diabetes remission occurs when blood sugar levels are in the normal range without medication for an extended period."),
    ("Can bariatric surgery reverse Type 2 diabetes?", "Bariatric surgery can help some people with obesity achieve diabetes remission by improving insulin sensitivity."),
    ("What is the future of diabetes treatment?", "Emerging treatments include gene therapy, beta cell regeneration, and advanced closed-loop insulin delivery systems."),
    ("What is stem cell therapy for diabetes?", "Stem cell therapy aims to regenerate insulin-producing beta cells in the pancreas."),
    ("What are the latest diabetes research breakthroughs?", "Breakthroughs include artificial pancreas systems, improved insulin formulations, and personalized medicine."),
    
    # Miscellaneous and Rare Cases
    ("Can pets get diabetes?", "Yes, dogs and cats can develop diabetes, requiring insulin therapy and diet changes."),
    ("What is brittle diabetes?", "Brittle diabetes is a rare form of Type 1 diabetes characterized by severe blood sugar swings."),
    ("How does climate affect diabetes?", "Extreme heat or cold can impact insulin absorption and blood sugar levels, requiring careful management."),
    ("What is reactive hypoglycemia?", "Reactive hypoglycemia occurs when blood sugar drops too low after a meal, often in people with prediabetes."),
    ("Can people with diabetes donate blood?", "Yes, diabetics can donate blood if their condition is well-controlled and they meet donation criteria."),
    ("Can a ketogenic diet help with diabetes?", "A low-carb ketogenic diet may help manage Type 2 diabetes, but it should be supervised by healthcare providers.")
]


# Sigmoid function ensures valid probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_diabetes_probability(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    # Coefficients from my linear regression table
    
    # Y=−0.8623+0.0141⋅(gender)+0.00138⋅(age)+0.0952⋅(hypertension)+0.1174⋅(heart disease)−0.0027⋅(smoking history)+0.00419⋅(bmi)+0.0812⋅(HbA1c level)+0.00227⋅(blood glucose level)
    # Where:
    # gender: Female = 0, Male = 1
    # hypertension and heart disease: No = 0, Yes = 1
    # smoking history: Never = 0, Former = 1, Not Current = 2, Current = 3, No Info = 4
    # age, bmi, HbA1c level, and blood glucose level: Continuous numeric values.

    intercept = -0.8623
    coef_gender = 0.0141
    coef_age = 0.00138
    coef_hypertension = 0.0952
    coef_heart_disease = 0.1174
    coef_smoking_history = -0.0027
    coef_bmi = 0.00419
    coef_HbA1c_level = 0.0812
    coef_blood_glucose = 0.00227
    
    # Linear regression output (z)
    z = (intercept
         + coef_gender * gender
         + coef_age * age
         + coef_hypertension * hypertension
         + coef_heart_disease * heart_disease
         + coef_smoking_history * smoking_history
         + coef_bmi * bmi
         + coef_HbA1c_level * HbA1c_level
         + coef_blood_glucose * blood_glucose_level)
    
    # Convert to probability using sigmoid
    probability = sigmoid(z)
    return probability


# Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)
    


def chatbot(chatbot_name, user_name):
    
    preprocessed_dataset = [(preprocess_text(question), response) for question, response in dataset]

    # Create training data
    X_train = [question for question, _ in preprocessed_dataset]
    y_train = [response for _, response in preprocessed_dataset]

    # Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)

    # Train the model (Support Vector Machine)
    model = SVC(probability=True) # Use the SVC classifier
    model.fit(X_train_vectors, y_train)

    # Evaluate the model
    X_train_vectors = vectorizer.transform(X_train)
    train_accuracy = model.score(X_train_vectors, y_train)
    
    print(chatbot_name + ": Hello " + user_name + "! I'm here to help you with any questions you have about diabetes. Whether it's about symptoms, treatments, prevention, or lifestyle tips—just ask, and I'll do my best to provide the answers you need!\n")



    while True:
        user_input = input(user_name + ": ")
        preprocessed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([preprocessed_input])
        predicted_response = model.predict(input_vector)[0]

        # Calculate cosine similarity for confidence level
        similarities = cosine_similarity(input_vector, X_train_vectors)
        max_similarity = similarities.max()
        confidence_level = max_similarity.item()
        # print(confidence_level)

        if confidence_level < 0.4:
            print(chatbot_name + ": As an AI, I don't have an answer for that. Please send an email to support@example.com")
        else:
            print(chatbot_name + ":", predicted_response)
        
        additional_question = input(chatbot_name + ": Is there anything else you want to know, (yes/no)?\n" + user_name + ": ")
        if additional_question.lower() == "no":
            print(chatbot_name + ": Okay, " + user_name + ", have a great day!")
            # Ask for user feedback
            feedback = input(chatbot_name + ": Could you please provide your valuable feedback on this conversation?\n" + user_name + ": ")
            # Sentiment analysis
            sentiment = TextBlob(feedback).sentiment
            polarity = sentiment.polarity
            if polarity > 0:
                print(chatbot_name + ": I'm glad you found my response helpful, " + user_name + "!")
                print("\nDisclaimer: This prediction is generated by a statistical model and may not always be accurate. "
                        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
                break
            elif polarity < 0:
                print(chatbot_name + ": I apologize if my response was not satisfactory, " + user_name + " could you please suggest improvements?")
                feedback_input = input(user_name + ": ")
                print(chatbot_name + ": Thank you for your feedback!")
                print("\nDisclaimer: This prediction is generated by a statistical model and may not always be accurate. "
                        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
                break
            else:
                print(chatbot_name + ": Thank you for your feedback, " + user_name + "!")
                print("\nDisclaimer: This prediction is generated by a statistical model and may not always be accurate. "
                        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
                break
        elif additional_question.lower() == "yes":
            continue



def main():
    user_name = input("Please enter your name: ")
    chatbot_name = "SugarSage"
    print("Hi", user_name, "my name is", chatbot_name, "and as your nurse I would like to predict your chances of diabetes.")
    print("To begin with, please go ahead and enter the following for me:")

    gender = int(input("Gender (Female = 0, Male = 1): "))
    age = int(input("Age: "))
    hypertension = int(input("Hypertension (No = 0, Yes = 1): "))
    heart_disease = int(input("Heart Disease (No = 0, Yes = 1): "))
    smoking_history = int(input("Smoking History (Never = 0, Former = 1, Not Current = 2, Current = 3, No Info = 4): "))
    bmi = float(input("BMI: "))
    HbA1c_level = float(input("HbA1c level: "))
    blood_glucose_level = float(input("Blood glucose level: "))

    prob = predict_diabetes_probability(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
    print(f"\nPredicted Probability of Diabetes: {prob:.2%}\n")

    if prob > 85:
        print("Based on this model's prediction, there is a high likelihood of diabetes. However, this is only an estimation. "
            "It is strongly recommended to consult a healthcare professional for an accurate diagnosis and personalized medical advice.\n")

    
    
    print("Now I will try to answer any questions you may have regarding Diabetes\n")

    chatbot(chatbot_name, user_name)


if __name__ == '__main__':
    main()