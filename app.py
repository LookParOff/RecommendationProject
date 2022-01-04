from flask import Flask, render_template, request, flash
import os
app = Flask(__name__)


@app.route("/")
def index():
	return render_template("index.html")


if os.environ.get("ON_HEROKU"):
	port = os.environ.get("PORT")
	app.run(port=port, debug=True)
else:
	app.run(host="127.0.0.1", port=5000, debug=True)
