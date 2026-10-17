
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the text vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Convert message into numerical features
            message_vector = vectorizer.transform([message])

            # Predict the message category
            result = model.predict(message_vector)[0]

            if result == 1:
                prediction = "SPAM MESSAGE"
            else:
                prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)

