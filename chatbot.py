import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV data
data = pd.read_csv("data.csv")

questions = data["Question"]
answers = data["Answer"]

# Convert questions into vectors
cv = CountVectorizer()
matrix = cv.fit_transform(questions)

print("College Chatbot Started")
print("Type 'exit' to stop")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye")
        break

    # Convert user input into vector
    user_vector = cv.transform([user_input])

    # Compare similarity
    similarity = cosine_similarity(user_vector, matrix)

    # Find best matching question
    index = similarity.argmax()

    print("Chatbot:", answers[index])