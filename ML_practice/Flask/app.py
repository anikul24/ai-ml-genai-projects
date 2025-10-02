from flask import Flask,render_template
## create WSGI instance

app = Flask(__name__)

@app.route("/")
def start():
    return "<html><h1>Gaggi is going to FLL next time!!!</h1></html>"


@app.route("/index")
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug= True)