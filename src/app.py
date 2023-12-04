import configparser

from flask import Flask, render_template, request, jsonify
from src import plot

app = Flask(__name__)

DEFAULT_CONFIG_PATH = 'default_config.cfg'


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
    config_path = request.form.get('configPath', DEFAULT_CONFIG_PATH)
    if request.method == 'POST':
        config_data = {
            'folderpath': request.form['folderPath'],
            'filepath': request.form['filePath'],
            'optionselect': request.form['optionSelect'],
            'numberinput': request.form['numberInput'],
            'textinput': request.form['textInput']
        }
        save_config(config_data, config_path)
        # Redirect or show a success message
    else:
        config_data = load_config(DEFAULT_CONFIG_PATH)
    # Render the form with config_data
    return render_template('config.html', config=config_data, default_config_path=config_path)


def save_config(config_data, config_path):
    if config_path == "":
        # throw warning
        return
    config = configparser.ConfigParser()
    config['DEFAULT'] = config_data
    with open(config_path, 'w') as configfile:
        config.write(configfile)


@app.route('/load_config', methods=['POST'])
def handle_load_config():
    config_path = request.form.get('configPath', DEFAULT_CONFIG_PATH)
    if config_path == "":
        return render_template('config.html', warning="No config file selected", default_config_path=DEFAULT_CONFIG_PATH)
    config_data = load_config(config_path)
    # You might want to pass this config_data to render in the config form
    return render_template('config.html', config=config_data, default_config_path=DEFAULT_CONFIG_PATH)


def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return dict(config['DEFAULT'])  # Assuming 'DEFAULT' section in config file


if __name__ == '__main__':
    app.run(debug=True)
