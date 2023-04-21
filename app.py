from flask import Flask, render_template, request

app = Flask(__name__)

def local_run():
    print("local")

def cluster_run():
    print("cluster")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_python', methods=['GET', 'POST'])
def run_python():
    # Your Python code here
    return "Python function executed!"

@app.route('/run_python2', methods=['POST'])
def run_python2():
    data = request.json
    run_location = data['run_location']

    if run_location == 'local':
        local_run()
    elif run_location == 'cluster':
        cluster_run()

    return "Python script has been executed"


if __name__ == '__main__':
    app.run(debug=True)
