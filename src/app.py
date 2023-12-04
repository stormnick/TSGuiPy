import configparser

from flask import Flask, render_template, request, jsonify
from src import plot

app = Flask(__name__)

def local_run():
    print("local")

def cluster_run():
    print("cluster")

@app.route('/')
def index():
    return render_template('home_page.html')

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

@app.route('/get_plot')
def get_plot():
    fig = plot.create_plot()
    return jsonify({"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"]})

@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        config_data = {
            'folderPath': request.form['folderPath'],
            'filePath': request.form['filePath'],
            'optionSelect': request.form['optionSelect'],
            'numberInput': request.form['numberInput'],
            'textInput': request.form['textInput']
        }
        save_config(config_data)
        # Redirect or show a success message
    else:
        config_data = load_config()
        # Render the form with config_data
    return render_template('config.html', config=config_data)

def save_config(config_data):
    config = configparser.ConfigParser()
    config['DEFAULT'] = config_data
    with open('config.cfg', 'w') as configfile:
        config.write(configfile)

def load_config():
    config = configparser.ConfigParser()
    config.read('config.cfg')
    return config['DEFAULT']


if __name__ == '__main__':
    app.run(debug=True)
