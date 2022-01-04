from flask import Flask, render_template, request, flash
import os
app = Flask(__name__)


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/recs", methods=['POST'])
def recs():
	print("I in recs. Ready to catch your request")
	input_json = request.get_json()  # force=True
	print("I catch. And I store request, here it is:")
	print(input_json)
	print("END!")
	return input_json


# if os.environ.get("ON_HEROKU"):
# 	port = os.environ.get("PORT")
# 	app.run(port=int(port), debug=True)
# else:
# 	app.run(host="127.0.0.1", port=5000, debug=True)
