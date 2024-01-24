import configparser
import importlib
import sys

from flask import Flask, render_template, request, jsonify
from TSGuiPy.src import plot
from plotting_tools.scripts_for_plotting import plot_synthetic_data_m3dis

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
        config_data = {}
        config_data['turbospectrum_compiler'] = {'compiler': request.form['turbospectrum_compiler__compiler']}
        # here are the other sections
        #"turbospectrum_path"] = self.old_turbospectrum_global_path
        #"interpolators_path"] = self.interpolators_path
        #"line_list_path"] = self.line_list_path
        #"model_atmosphere_grid_path_1d"] = self.model_atmosphere_grid_path_1d
        #"model_atmosphere_grid_path_3d"] = self.model_atmosphere_grid_path_3d
        #"model_atoms_path"] = self.model_atoms_path
        #"departure_file_path"] = self.departure_file_path
        #"departure_file_config_path"] = self.departure_file_config_path
        #"output_path"] = self.old_output_folder_path_global
        #"linemasks_path"] = self.linemasks_path
        #"spectra_input_path"] = self.spectra_input_path
        #"fitlist_input_path"] = self.fitlist_input_path
        #"temporary_directory_path"] = self.old_global_temporary_directory
        config_data['MainPaths'] = {
            'turbospectrum_path': request.form['MainPaths__turbospectrum_path'],
            'interpolators_path': request.form['MainPaths__interpolators_path'],
            'line_list_path': request.form['MainPaths__line_list_path'],
            'model_atmosphere_grid_path_1d': request.form['MainPaths__model_atmosphere_grid_path_1d'],
            'model_atmosphere_grid_path_3d': request.form['MainPaths__model_atmosphere_grid_path_3d'],
            'model_atoms_path': request.form['MainPaths__model_atoms_path'],
            'departure_file_path': request.form['MainPaths__departure_file_path'],
            'departure_file_config_path': request.form['MainPaths__departure_file_config_path'],
            'output_path': request.form['MainPaths__output_path'],
            'linemasks_path': request.form['MainPaths__linemasks_path'],
            'spectra_input_path': request.form['MainPaths__spectra_input_path'],
            'fitlist_input_path': request.form['MainPaths__fitlist_input_path'],
            'temporary_directory_path': request.form['MainPaths__temporary_directory_path']

        }
        save_config(config_data, config_path)
        config_data = load_config(config_path)
        # Redirect or show a success message
    else:
        config_data = load_config(DEFAULT_CONFIG_PATH)
        config_data2 = load_config(config_path)
    # Render the form with config_data
    return render_template('config.html', config=config_data, default_config_path=config_path, config_data=config_data2)


def save_config(config_data, config_path):
    if config_path == "":
        # throw warning
        return
    config = configparser.ConfigParser()
    for section in config_data:
        config[section] = config_data[section]
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
    # convert config to dict
    config_dict = {}
    for section in config.sections():
        for key in config[section]:
            new_key = f"{section}__{key}"
            config_dict[new_key] = config[section][key]
    return config_dict

def import_module_from_path(module_name, file_path):
    """
    Dynamically imports a module or package from a given file path.

    Parameters:
    module_name (str): The name to assign to the module.
    file_path (str): The file path to the module or package.

    Returns:
    module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Module spec not found for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

@app.route('/results')
def results():
    return render_template('index.html')


@app.route('/generate_synthetic_spectrum')
def generate_synthetic_spectrum():
    return render_template('generate_synthetic_spectrum.html')

def call_m3d(teff, logg, feh, lmin, lmax):
    m3dis_paths = {"m3dis_path": "/Users/storm/PycharmProjects/3d_nlte_stuff/m3dis_l/m3dis/experiments/Multi3D/",
                   "nlte_config_path": "/Users/storm/docker_common_folder/TSFitPy/input_files/nlte_data/nlte_filenames.cfg",
                   "model_atom_path": "/Users/storm/docker_common_folder/TSFitPy/input_files/nlte_data/model_atoms/",
                   "model_atmosphere_grid_path": "/Users/storm/docker_common_folder/TSFitPy/input_files/model_atmospheres/",
                   "line_list_path": "/Users/storm/docker_common_folder/TSFitPy/input_files/linelists/linelist_for_fitting/",
                   "3D_atmosphere_path": None}  # change to path to 3D atmosphere if running 3D model atmosphere

    atmosphere_type = "1D"  # "1D" or "3D"
    hash_table_size = 100
    n_nu = 1
    mpi_cores = 1
    # if 3D atmosphere is used, then the following is needed
    dims = 23  # dimensions of the atmosphere
    atmos_format = 'Multi'  # MUST, Stagger or Multi
    nx = 10  # only if Stagger
    ny = 10  # only if Stagger
    nz = 230  # only if Stagger
    snap = 1  # snapshot number, only if Stagger
    # nlte settings
    nlte_flag = False
    # nlte settings, if nlte_flag = False, these are not used
    nlte_iterations_max = 10
    nlte_convergence_limit = 0.001
    element_in_nlte = "Fe"  # can choose only one element
    element_abundances = {}
    wavelength, norm_flux = plot_synthetic_data_m3dis(m3dis_paths, teff, logg, feh, 1.0, lmin, lmax, 0.01, atmosphere_type, atmos_format, n_nu, mpi_cores, hash_table_size, nlte_flag, element_in_nlte, element_abundances, snap, dims, nx, ny, nz, nlte_iterations_max, nlte_convergence_limit, m3dis_package_name="m3dis_mine")
    return list(wavelength), list(norm_flux)


@app.route('/get_m3d_plot')
def get_plot_m3d():
    teff = request.args.get('teff', type=float)
    logg = request.args.get('logg', type=float)
    feh = request.args.get('feh', type=float)
    lmin = request.args.get('lmin', type=float)
    lmax = request.args.get('lmax', type=float)
    print("get_plot_m3d")
    wavelength, flux = call_m3d(teff, logg, feh, lmin, lmax)
    fig = plot.create_plot_data(wavelength, flux)
    return jsonify({"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"]})

if __name__ == '__main__':
    app.run(debug=True)
    #generate_synthetic_spectrum()
