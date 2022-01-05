from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/recs", methods=['POST'])
def recs():
	# TODO input: json of ratings of man. output: json of recommended books, like jsons for Andrew
	print("I in recs. Ready to catch your request")
	input_json = request.get_json()  # force=True
	print("I catch. And I store request, here it is:")
	print(input_json)
	print("END!")
	return input_json
