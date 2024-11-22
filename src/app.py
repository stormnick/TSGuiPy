from __future__ import annotations

import sys
import os

import scipy
from alembic.command import current

# This is a workaround to add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',)))

import configparser
import json
import tempfile
import zipfile
import numpy as np
import plotly
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from TSGuiPy.src import plot
from plotting_tools.scripts_for_plotting import plot_synthetic_data_m3dis, load_output_data, plot_synthetic_data, get_just_parsed_linelist
from scripts.auxiliary_functions import apply_doppler_correction, calculate_equivalent_width
from scripts.solar_abundances import periodic_table
import io
import csv
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 50 Megabytes

DEFAULT_CONFIG_PATH = 'default_config.cfg'
data_results_storage = {'fitted_spectra': [], "options": [], "linemask_center_wavelengths": [], "observed_spectra": {},
                        "observed_spectra_synthetic": {}, "synthetic_spectra": {"wavelength": [], "flux": []},
                        "currently_plotted_spectra": None}

CONFIG_WITH_DEFAULT_PATHS = "default_paths.cfg"
default_paths = configparser.ConfigParser()
default_paths.read(CONFIG_WITH_DEFAULT_PATHS)
default_paths = default_paths["PathsToTheCodes"]
default_name_new_flags = "new_flags.csv"

"""
HOME PAGE
"""

@app.route('/')
def render_html_main_page():
    return render_template('home_page.html')

"""
CONFIG CREATOR
"""

@app.route('/config', methods=['GET', 'POST'])
def render_html_config():
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

"""
FITTED SPECTRA RESULTS VIEWER
"""

@app.route('/spectral_fits_viewer')
def render_html_spectral_fits_viewer():
    return render_template('spectral_fits_viewer.html', options=data_results_storage["options"], options_linemask=data_results_storage["linemask_center_wavelengths"])


@app.route('/get_plot_fitted_spectra_one_star', methods=['POST'])
def get_plot_fitted_result_one_star():
    #print("plot_fitted_result")
    data = request.json
    specname = data['specname']
    overplot_synthetic_data = data['overplotBlendsCheck']
    linelist_path = data['linelistPath']
    loggf_limit = float(data['loggf_limit'])
    includeWarnings = data['includeWarnings']
    auto_save_new_errors = data['autoSaveNewErrors']
    prev_spectra_checked_boxes = data['checkedBoxes']

    file_new_errors = os.path.join("./", default_name_new_flags)
    if os.path.exists(file_new_errors):
        current_new_errors = pd.read_csv(file_new_errors)
    else:
        current_new_errors = pd.DataFrame(columns=["specname", "linemask", "extra_error"])

    prev_spectra_name = data_results_storage['currently_plotted_spectra']
    if auto_save_new_errors and prev_spectra_name is not None:
        for one_linemask in prev_spectra_checked_boxes:
            wavelength = one_linemask['id']
            if wavelength != "checkbox0":
                checked = one_linemask['checked']
                if not checked:
                    extra_error_value = 1
                else:
                    extra_error_value = 0
                # Check if a row with the same "specname" and "linemask" exists
                mask = (
                    (current_new_errors['specname'] == prev_spectra_name) &
                    (current_new_errors['linemask'].astype(str) == wavelength)
                )

                if mask.any():
                    # Update 'extra_error' in the existing row(s)
                    current_new_errors.loc[mask, 'extra_error'] = extra_error_value
                else:
                    # Append a new row as before
                    current_new_errors.loc[len(current_new_errors)] = {
                        "specname": prev_spectra_name,
                        "linemask": wavelength,
                        "extra_error": extra_error_value
                    }

        current_new_errors.to_csv(file_new_errors, index=False)

    data_results_storage['currently_plotted_spectra'] = specname

    figures = []
    avg_feh = []
    avg_vmic = []
    checked_boxes = []
    center_wavelengths_all = []
    for linemask_idx, linemask_center_wavelength in enumerate(data_results_storage["linemask_center_wavelengths"]):
        avg_feh.append(data_results_storage['fitted_spectra'][specname]['Fe_H'][linemask_idx])
        avg_vmic.append(data_results_storage['fitted_spectra'][specname]['vmic'][linemask_idx])

        center_wavelengths = data_results_storage["linemask_center_wavelengths"][linemask_idx]
        center_wavelengths_all.append(center_wavelengths)
        left_wavelengths = data_results_storage["linemask_left_wavelengths"][linemask_idx]
        right_wavelengths = data_results_storage["linemask_right_wavelengths"][linemask_idx]
        # load spectra
        wavelength_fitted = data_results_storage['fitted_spectra'][specname]["wavelength_fitted"]
        flux_fitted = data_results_storage['fitted_spectra'][specname]["flux_fitted"]
        wavelength_observed = data_results_storage['fitted_spectra'][specname]["wavelength_observed"]
        flux_observed = data_results_storage['fitted_spectra'][specname]["flux_observed"]
        rv_correction = data_results_storage['fitted_spectra'][specname]['spectra_rv']
        wavelength_just_blend = data_results_storage['fitted_spectra'][specname]["wavelength_just_blend"]
        flux_just_blend = data_results_storage['fitted_spectra'][specname]["flux_just_blend"]
        wavelength_minus_sensitivity = data_results_storage['fitted_spectra'][specname]["wavelength_minus_sensitivity"]
        flux_minus_sensitivity = data_results_storage['fitted_spectra'][specname]["flux_minus_sensitivity"]
        wavelength_plus_sensitivity = data_results_storage['fitted_spectra'][specname]["wavelength_plus_sensitivity"]
        flux_plus_sensitivity = data_results_storage['fitted_spectra'][specname]["flux_plus_sensitivity"]
        # ew_just_line	ew_blend	ew_sensitivity
        ew_just_line = data_results_storage['fitted_spectra'][specname]["ew_just_line"][linemask_idx]
        ew_blend = data_results_storage['fitted_spectra'][specname]["ew_blend"][linemask_idx]
        ew_sensitivity = data_results_storage['fitted_spectra'][specname]["ew_sensitivity"][linemask_idx]
        # apply rv correction
        rv_fitted = data_results_storage['fitted_spectra'][specname]['fitted_rv'][linemask_idx]
        wavelength_observed_rv = (apply_doppler_correction(wavelength_observed, rv_correction + rv_fitted))

        if overplot_synthetic_data and (data_results_storage["fitting_method"] == "lbl" or data_results_storage["fitting_method"] == "vmic"):
            parsed_linelist_info = get_just_parsed_linelist(left_wavelengths - 0.5, right_wavelengths + 0.5, linelist_path, loggf_limit)

            parsed_linelist_dict = []
            for i, (wavelength_element, element_linelist, loggf) in enumerate(parsed_linelist_info):
                parsed_linelist_dict.append(
                    {"wavelength": wavelength_element, "element": element_linelist, "loggf": loggf,
                     "name": f"{wavelength_element:.2f} {element_linelist} {loggf:.3f}"})


        else:
            parsed_linelist_dict = []


        fitted_value = data_results_storage['fitted_spectra'][specname]['fitted_value'][linemask_idx]
        title = (f"{data_results_storage['fitted_value_label']} = {fitted_value:.2f}, rv_fit = {rv_fitted:.2f}, "
                 f"vmac = {data_results_storage['fitted_spectra'][specname]['vmac'][linemask_idx]:.2f}, vsini = {data_results_storage['fitted_spectra'][specname]['rotation'][linemask_idx]:.2f}<br>"
                 f"ERR = {data_results_storage['fitted_spectra'][specname]['flag_error'][linemask_idx]}, WARN = {data_results_storage['fitted_spectra'][specname]['flag_warning'][linemask_idx]}, chisqr = {data_results_storage['fitted_spectra'][specname]['chi_squared'][linemask_idx]:.6f}<br>"
                 f"EW: total = {data_results_storage['fitted_spectra'][specname]['ew'][linemask_idx]:.2f}, line = {ew_just_line:.2f}, blend = {ew_blend:.2f}, sensit = {ew_sensitivity:.2f}")
        # for wavelength_fitted if difference between the points is lower than ldelta, remove any points in-between such that the difference is higher than ldelta
        # in general difference is constant
        filtered_wavelengths = wavelength_fitted # Start with the first element
        filtered_flux = flux_fitted

        fig = plot.create_plot_data_one_star(np.asarray(filtered_wavelengths), np.asarray(filtered_flux), wavelength_observed_rv,
                                             flux_observed, left_wavelengths, right_wavelengths, center_wavelengths, title, wavelength_just_blend,
                                             flux_just_blend, wavelength_plus_sensitivity, flux_plus_sensitivity,
                                             wavelength_minus_sensitivity, flux_minus_sensitivity)
        figure_data = {
            "figure": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            "value": fitted_value,
            "columns": parsed_linelist_dict
        }
        figures.append(figure_data)

        mask = (
                (current_new_errors['specname'] == specname) &
                (current_new_errors['linemask'].astype(str) == str(center_wavelengths))
        )

        if mask.any():
            extra_error_value = current_new_errors.loc[mask, 'extra_error'].values[0]
        else:
            extra_error_value = 0

        if extra_error_value == 0:
            if data_results_storage['fitted_spectra'][specname]['flag_error'][linemask_idx] != 0:
                checked_boxes.append(False)
            elif data_results_storage['fitted_spectra'][specname]['flag_warning'][linemask_idx] != 0 and not includeWarnings:
                checked_boxes.append(False)
            else:
                checked_boxes.append(True)
        else:
            checked_boxes.append(False)

    stellar_param_teff = data_results_storage['fitted_spectra'][specname]['teff']
    stellar_param_logg = data_results_storage['fitted_spectra'][specname]['logg']
    stellar_param_feh = np.average(avg_feh)
    stellar_param_vmic = np.average(avg_vmic)

    return jsonify(figures=figures, stellar_param=f"{int(stellar_param_teff)}/{stellar_param_logg:.2f}/{stellar_param_feh:.2f}/{stellar_param_vmic:.2f}",
                   checked_boxes=checked_boxes, center_wavelengths_all=center_wavelengths_all)

"""
ABUNDANCE DIAGRAM
"""

@app.route('/abundance_diagram')
def render_html_abundance_diagram():
    return render_template('analyse_abundance_diagram.html')

@app.route('/get_plot_abundance_diagram', methods=['POST'])
def plot_abundance_diagram():
    data = request.json
    remove_errors = data['removeErrorsBool']
    remove_warnings = data['removeWarningsBool']
    chisqr_limit = float(data['chisqrLimit'])

    specnames = data_results_storage['fitted_spectra'].keys()

    # get the data from the fitted spectra
    x_values = []
    y_values = []

    # so we want to get average of the fitted value for each specname, within the chisqr_limit and without errors and warnings
    for specname in specnames:
        # get the indices of the values that are within the chisqr_limit, and without errors and warnings
        #indices = np.where((data_results_storage['fitted_spectra'][specname]['chi_squared'] <= chisqr_limit) & (data_results_storage['fitted_spectra'][specname]['flag_error'] == 0) & (data_results_storage['fitted_spectra'][specname]['flag_warning'] == 0))
        # only do above if remove_errors and remove_warnings are True
        chi_sqr_values = np.asarray(list(data_results_storage['fitted_spectra'][specname]['chi_squared'].values()))
        flag_error_values = np.asarray(list(data_results_storage['fitted_spectra'][specname]['flag_error'].values()))
        flag_warning_values = np.asarray(list(data_results_storage['fitted_spectra'][specname]['flag_warning'].values()))
        if remove_errors and remove_warnings:
            indices = np.where((chi_sqr_values <= chisqr_limit) & (flag_error_values == 0) & (flag_warning_values == 0))[0]
        elif remove_errors:
            indices = np.where((chi_sqr_values <= chisqr_limit) & (flag_error_values == 0))[0]
        elif remove_warnings:
            indices = np.where((chi_sqr_values <= chisqr_limit) & (flag_warning_values == 0))[0]
        else:
            indices = np.where(chi_sqr_values <= chisqr_limit)[0]

        if np.size(indices) > 0:
            x_value_temp = []
            y_value_temp = []
            for index in indices:
                x_value_temp.append(np.average(data_results_storage['fitted_spectra'][specname]['Fe_H'][index]))
                y_value_temp.append(np.average(data_results_storage['fitted_spectra'][specname]['fitted_value'][index]))

            x_values.append(np.average(x_value_temp))
            y_values.append(np.average(y_value_temp))

    fitted_element_name = data_results_storage['fitted_value_label']

    fig = plot.plot_abundance_plot(x_values, y_values, f"[Fe/H]", fitted_element_name, f"Abundance Diagram, stars: {len(x_values)}")
    return jsonify({"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"]})


"""
GENERATE SYNTHETIC SPECTRUM
"""

@app.route('/generate_synthetic_spectrum')
def render_html_generate_synthetic_spectrum():
    return render_template('generate_synthetic_spectrum.html')

def degrade_to_snr(flux, snr):
    """
    :param wavelength:
    :param flux:
    :param snr:
    :return: degraded flux
    """
    flux = np.asarray(flux)
    # Calculate the standard deviation of the noise for each flux point
    sigma = flux / snr

    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, size=flux.shape)

    # Add noise to the original flux to create the degraded spectrum
    degraded_flux = flux + noise

    return list(degraded_flux)

def resample_spectrum(wavelengths, flux, sampling_rate_AA):
    """
    Resample a spectrum to a new wavelength grid with a specified sampling rate in milliangstroms.

    Parameters:
    - wavelengths (array-like): The original wavelengths in angstroms.
    - flux (array-like): The original flux values.
    - sampling_rate_AA (float): The desired sampling rate in milliangstroms.

    Returns:
    - tuple: A tuple containing the new wavelengths and the resampled flux values.
    """
    new_wavelength_start, new_wavelength_end = np.min(wavelengths), np.max(wavelengths)

    # Create the new wavelength grid
    new_wavelengths = np.arange(new_wavelength_start, new_wavelength_end, sampling_rate_AA)

    # Create the interpolation function
    interp_func = scipy.interpolate.interp1d(wavelengths, flux, kind='linear', fill_value=1, bounds_error=False)

    # Interpolate the flux to the new wavelength grid
    new_flux = interp_func(new_wavelengths)

    return new_wavelengths, new_flux


@app.route('/get_plot_synthetic_spectrum', methods=['POST'])
def get_plot_synthetic_spectrum():
    data = request.json
    teff = float(data['teff'])
    logg = float(data['logg'])
    feh = float(data['feh'])
    vmic = float(data['vmic'])
    lmin = float(data['lmin'])
    lmax = float(data['lmax'])
    ldelta = float(data['deltal'])
    nlte_element = data['nlte_element']
    nlte_iter = int(data['nlte_iter'])
    xfeabundances = data['m3d_xfeabundances']
    vmac = float(data['vmac'])
    resolution = float(data['resolution'])
    rotation = float(data['rotation'])
    obs_rv = float(data['obs_rv'])
    loggf_limit = float(data['loggf_limit'])
    linelist_path = data['linelist_path']
    code_type = data['code_type']
    synthesise_molecules = data['synthesiseMolecules'],
    snr = float(data['snr'])
    #print("get_plot_m3d")

    element_abundances = {}

    xfeabundances = xfeabundances.split("\n")
    for xfeabundance in xfeabundances:
        if xfeabundance != "":
            element, abundance = xfeabundance.split(" ")
            # convert element to element name
            # try to convert to int
            try:
                element = int(element)
                element_name = periodic_table[element]
            except ValueError:
                # if it fails, it is a string
                element_name = element.lower().capitalize()
            element_abundances[element_name] = float(abundance)

    if code_type.lower() == 'm3d':
        wavelength, flux, parsed_linelist_info = call_m3d(teff, logg, feh, vmic, lmin, lmax, ldelta, nlte_element,
                                                          nlte_iter, element_abundances, vmac, rotation, resolution,
                                                          loggf_limit=loggf_limit, linelist_path=linelist_path)
    elif code_type.lower() == 'ts':
        wavelength, flux, parsed_linelist_info = call_ts(teff, logg, feh, vmic, lmin, lmax, ldelta, nlte_element,
                                                          element_abundances, vmac, rotation, resolution,
                                                          loggf_limit=loggf_limit, linelist_path=linelist_path,
                                                          synthesise_molecules=synthesise_molecules)

    parsed_linelist_dict = []
    for i, (wavelength_element, element_linelist, loggf) in enumerate(parsed_linelist_info):
        parsed_linelist_dict.append({"wavelength": wavelength_element, "element": element_linelist, "loggf": loggf,
                                     "name": f"{wavelength_element:.2f} {element_linelist} {loggf:.3f}"})

    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)

    # we need to resample back to delta lambda
    wavelength, flux = resample_spectrum(wavelength, flux, ldelta)

    if snr != 0:
        flux = degrade_to_snr(flux, snr)

    if data_results_storage["observed_spectra"]:
        wavelength_observed = data_results_storage['observed_spectra']["wavelength"]
        flux_observed = data_results_storage['observed_spectra']["flux"]
        wavelength_observed = np.asarray(wavelength_observed)
        wavelength_observed = apply_doppler_correction(wavelength_observed, obs_rv)

        # cut the observed spectra to the same range as the synthetic spectra
        wavelength_observed, flux_observed = zip(*[(wavelength_observed[i], flux_observed[i]) for i in range(len(wavelength_observed)) if lmin - 2 <= wavelength_observed[i] <= lmax + 2])
        # convert to lists
        wavelength_observed, flux_observed = list(wavelength_observed), list(flux_observed)
    else:
        wavelength_observed, flux_observed = [], []
    fig = plot.plot_synthetic_data(wavelength, flux, lmin, lmax, wavelength_observed, flux_observed)
    data_results_storage['synthetic_spectra']["wavelength"] = list(wavelength)
    data_results_storage['synthetic_spectra']["flux"] = list(flux)
    return jsonify({"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"], "columns": parsed_linelist_dict})

@app.route('/download_data', methods=['POST'])
def download_data():
    data = request.form.get('data')
    data = json.loads(data)
    teff = int(data.get('teff'))
    logg = round(float(data.get('logg')), 2)
    feh = round(float(data.get('feh')), 2)
    #vmic = round(float(data.get('vmic')), 2)
    #lmin = float(data.get('lmin'))
    #lmax = float(data.get('lmax'))
    #ldelta = float(data.get('deltal'))
    ##nlte_element = data['nlte_element']
    ##nlte_iter = int(data['nlte_iter'])
    ##xfeabundances = data['m3d_xfeabundances']
    #vmac = float(data.get('vmac'))
    #resolution = float(data.get('resolution'))
    #rotation = float(data.get('rotation'))
    ##obs_rv = float(data['obs_rv'])
    ##loggf_limit = float(data['loggf_limit'])
    ##linelist_path = data['linelist_path']
    #code_type = data.get('code_type')
    #synthesise_molecules = data.get('synthesiseMolecules')
    #snr = float(data.get('snr'))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['wavelength', 'norm_flux'])  # Header row
    # "synthetic_spectra": {"wavelength": [], "flux": []}
    wavelength = data_results_storage['synthetic_spectra']["wavelength"]
    flux = data_results_storage['synthetic_spectra']["flux"]
    # cut to 4 and 6 decimal places
    wavelength = [f"{wavelength[i]:.4f}" for i in range(len(wavelength))]
    flux = [f"{flux[i]:.6f}" for i in range(len(flux))]
    #writer.writerow(f"#teff={teff}, logg={logg}, feh={feh}, vmic={vmic}, lmin={lmin}, lmax={lmax}, ldelta={ldelta}, vmac={vmac}, resolution={resolution}, rotation={rotation}, code_type={code_type}, synthesise_molecules={synthesise_molecules}, snr={snr}".split(","))
    writer.writerows(zip(wavelength, flux))
    csv_data = output.getvalue()
    output.close()

    response = make_response(csv_data)
    response.headers['Content-Disposition'] = f'attachment; filename=synthetic_spectra_{teff}_{logg}_{feh}.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

"""
PLOT OBSERVED SPECTRA
"""

@app.route('/plot_observed_spectrum')
def render_html_plot_observed_spectrum():
    return render_template('plot_observed_spectra.html')

@app.route('/get_plot_observed_spectrum', methods=['POST'])
def get_plot_observed_spectra():
    data = request.json
    rv = float(data['obs_rv'])
    rv_synthetic = float(data['synthetic_rv'])

    wavelength_observed_rv_corrected = []
    wavelength_observed, flux_observed = [], []

    if data_results_storage['observed_spectra']:
        wavelength_observed, flux_observed = data_results_storage['observed_spectra']["wavelength"], data_results_storage['observed_spectra']["flux"]
        if np.size(wavelength_observed) > 0:
            wavelength_observed_rv_corrected = apply_doppler_correction(wavelength_observed, rv)


    wavelength_synthetic, flux_synthetic = [], []
    wavelength_synthetic_rv_corrected = []

    if data_results_storage['observed_spectra_synthetic']:
        wavelength_synthetic, flux_synthetic = data_results_storage['observed_spectra_synthetic']["wavelength"], data_results_storage['observed_spectra_synthetic']["flux"]
        if np.size(wavelength_synthetic) > 0:
            wavelength_synthetic_rv_corrected = apply_doppler_correction(wavelength_synthetic, rv_synthetic)




    fig = plot.plot_observed_spectra(wavelength_observed_rv_corrected, flux_observed, wavelength_synthetic_rv_corrected, flux_synthetic)
    return jsonify({"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"]})


"""
CALLING FORTRAN CODES TO GENERATE SPECTRA
"""

def call_m3d(teff, logg, feh, vmic, lmin, lmax, ldelta, nlte_element, nlte_iter, xfeabundances: dict, vmac, rotation, resolution, linelist_path=None, loggf_limit=-5.0):
    if linelist_path is None:
        linelist_path = default_paths["default_linelist_path"]
    m3dis_paths = {"m3dis_path": default_paths["m3dis_path"],
                   "nlte_config_path": default_paths["nlte_config_path"],
                   "model_atom_path": default_paths["model_atom_path"],
                   "model_atmosphere_grid_path": default_paths["model_atmosphere_grid_path"],
                   "line_list_path": linelist_path,
                   "3D_atmosphere_path": default_paths["3D_atmosphere_path"]}  # change to path to 3D atmosphere if running 3D model atmosphere

    atmosphere_type = "1D"  # "1D" or "3D"
    hash_table_size = 100
    n_nu = 16
    mpi_cores = 8
    # if 3D atmosphere is used, then the following is needed
    dims = 23  # dimensions of the atmosphere
    atmos_format = 'Multi'  # MUST, Stagger or Multi
    nx = 10  # only if Stagger
    ny = 10  # only if Stagger
    nz = 230  # only if Stagger
    snap = 1  # snapshot number, only if Stagger
    # nlte settings, if nlte_flag = False, these are not used
    nlte_iterations_max = nlte_iter
    nlte_convergence_limit = 0.00001
    if nlte_element != "none":
        element_in_nlte = nlte_element
        nlte_flag = True
    else:
        element_in_nlte = ""
        nlte_flag = False
    element_abundances = xfeabundances
    # convert xfeabundances to dictionary. first number is the element number in periodic table, second is the abundance. the separation between elements is \n

    wavelength, norm_flux, parsed_linelist_info = plot_synthetic_data_m3dis(m3dis_paths, teff, logg, feh, vmic, lmin, lmax, ldelta,
                                                      atmosphere_type, atmos_format, n_nu, mpi_cores, hash_table_size,
                                                      nlte_flag, element_in_nlte, element_abundances, snap, dims, nx, ny, nz,
                                                      nlte_iterations_max, nlte_convergence_limit, m3dis_package_name="m3dis",
                                                      verbose=False, macro=vmac, resolution=resolution, rotation=rotation,
                                                      return_parsed_linelist=True, loggf_limit_parsed_linelist=loggf_limit,
                                                      plot_output=False)

    return list(wavelength), list(norm_flux), parsed_linelist_info

def call_ts(teff, logg, feh, vmic, lmin, lmax, ldelta, nlte_element, xfeabundances: dict, vmac, rotation, resolution, linelist_path=None, loggf_limit=None, synthesise_molecules=False):
    if linelist_path is None:
        linelist_path = default_paths["default_linelist_path"]
    ts_paths = {"turbospec_path": default_paths["turbospectrum_path"],
                       "interpol_path": default_paths["interpol_path"],
                       "model_atom_path": default_paths["model_atom_path"],
                       "departure_file_path": default_paths["departure_file_path"],
                       "model_atmosphere_grid_path": default_paths["model_atmosphere_grid_path"],
                       "line_list_path": linelist_path}  # change to path to 3D atmosphere if running 3D model atmosphere

    atmosphere_type = "1D"  # "1D" or "3D"
    if nlte_element != "none":
        element_in_nlte = [nlte_element]
        nlte_flag = True
    else:
        element_in_nlte = [""]
        nlte_flag = False
    element_abundances = xfeabundances
    # convert xfeabundances to dictionary. first number is the element number in periodic table, second is the abundance. the separation between elements is \n

    wavelength, norm_flux, parsed_linelist_dict = plot_synthetic_data(ts_paths, teff, logg, feh, vmic, lmin, lmax, ldelta,
                                                atmosphere_type, nlte_flag, element_in_nlte, element_abundances, synthesise_molecules,
                                                verbose=False, macro=vmac, resolution=resolution, rotation=rotation, do_matplotlib_plot=False,
                                                return_parsed_linelist=True, loggf_limit_parsed_linelist=loggf_limit)

    return list(wavelength), list(norm_flux), parsed_linelist_dict


"""
CALCULATE EQUIVALENT WIDTH
"""

@app.route('/synthetic_calculate_integral', methods=['POST'])
def synthetic_calculate_integral():
    """left_x_boundary: document.getElementById('left_x_boundary').value,
        right_x_boundary: document.getElementById('right_x_boundary').value,
        plot_data: document.getElementById('plot-container').data"""
    data = request.json
    left_x_boundary = float(data['left_x_boundary'])
    right_x_boundary = float(data['right_x_boundary'])
    plot_data = data['plot_data']

    # if left_x_boundary > right_x_boundary, swap them
    if left_x_boundary > right_x_boundary:
        left_x_boundary, right_x_boundary = right_x_boundary, left_x_boundary

    # if left_x_boundary == right_x_boundary, return 0
    if left_x_boundary == right_x_boundary:
        return jsonify({"integral_synthetic": 0, "integral_observed": 0})

    trace = plot_data[0]
    if len(plot_data) > 1:
        trace_obs = plot_data[1]
        x_obs = np.asarray(trace_obs['x'])
        y_obs = np.asarray(trace_obs['y'])
    else:
        x_obs, y_obs = np.array([]), np.array([])

    x_fitted = np.asarray(trace['x'])
    y_fitted = np.asarray(trace['y'])


    # find the indices of wavelength that are within lmin and lmax
    indices_fitted = np.where((x_fitted >= left_x_boundary) & (x_fitted <= right_x_boundary))
    indices_observed = np.where((x_obs >= left_x_boundary) & (x_obs <= right_x_boundary))
    # calculate the integral
    # if the indices are empty, return 0
    if np.size(indices_fitted) == 0:
        ew_fitted = 0
    else:
        ew_fitted = calculate_equivalent_width(x_fitted[indices_fitted], y_fitted[indices_fitted], left_x_boundary, right_x_boundary) * 1000
    if np.size(indices_observed) == 0:
        ew_observed = 0
    else:
        ew_observed = calculate_equivalent_width(x_obs[indices_observed], y_obs[indices_observed], left_x_boundary, right_x_boundary) * 1000
    return jsonify({"integral_synthetic": round(ew_fitted, 3), "integral_observed": round(ew_observed, 3)})

"""
UPLOADING RESULTS FILES AND HANDLING THEM
"""

@app.route('/upload_zip', methods=['POST'])
def upload_zipped_file():
    if 'file' not in request.files:
        return redirect(url_for('plot_results'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('plot_results'))
    if file and file.filename.endswith('.zip'):
        filename = 'temp_uploaded_file.zip'
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            extract_and_process_zip(file_path)
        return jsonify({'message': 'Upload successful', 'status': 'success'})
    else:
        return jsonify({'message': 'Not zip', 'status': 'error'})

def extract_and_process_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as temp_dir_unzipped:
            zip_ref.extractall(temp_dir_unzipped)
            # Process files here
            data_results_storage['fitted_spectra'] = {}
            #for file_name in zip_ref.namelist():
            #    filepath = os.path.join(temp_dir_unzipped, file_name.filename.split("/")[1])
            #    file_name.save(filepath)
            # print all directories in temp_dir_unzipped
            dir_unzipped = os.path.join(temp_dir_unzipped, os.listdir(temp_dir_unzipped)[0])
            parsed_config_dict = load_output_data(dir_unzipped)
            data_results_storage["parsed_config_dict"] = parsed_config_dict
            process_file(dir_unzipped, data_results_storage["parsed_config_dict"])
            data_results_storage["options"] = data_results_storage["parsed_config_dict"]["specname_fitlist"]
                # now route to analyse_results

    os.remove(zip_path)

@app.route('/upload', methods=['POST'])
def upload_folder():
    files = request.files.getlist('folder')
    if not files:
        return 'No files uploaded'
    # Temporary directory to save uploaded files
    # create temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        #os.mkdir(temp_dir)
        # Save files and process them
        # find a file that ends with .cfg
        data_results_storage['fitted_spectra'] = {}
        for file in files:
            if "/" in file.filename:
                filepath = os.path.join(temp_dir, file.filename.split("/")[1])
            else:
                filepath = os.path.join(temp_dir, file.filename)
            file.save(filepath)
        parsed_config_dict = load_output_data(temp_dir)
        data_results_storage["parsed_config_dict"] = parsed_config_dict
        process_file(temp_dir, data_results_storage["parsed_config_dict"])
        data_results_storage["options"] = data_results_storage["parsed_config_dict"]["specname_fitlist"]
    #options = data_results_storage["options"]
    # Optionally, clean up by deleting the temporary files
    # now route to analyse_results
    return jsonify({'message': 'Upload successful', 'status': 'success'})

@app.route('/upload_observed_spectra', methods=['POST'])
def upload_spectra():
    # Get the uploaded files
    file = request.files['file']
    print(file)
    # Temporary directory to save uploaded files
    # create temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        if "/" in file.filename:
            filepath = os.path.join(temp_dir, file.filename.split("/")[1])
        else:
            filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)
        wavelength_observed, flux_observed = np.loadtxt(filepath, unpack=True, usecols=(0,1), dtype=float)
        # find any nans and remove them
        indices = np.where(np.isnan(flux_observed))
        wavelength_observed = np.delete(wavelength_observed, indices)
        flux_observed = np.delete(flux_observed, indices)
        data_results_storage['observed_spectra']["wavelength"] = wavelength_observed
        data_results_storage['observed_spectra']["flux"] = flux_observed
        #parsed_config_dict = load_output_data(temp_dir)
        #data_results_storage["parsed_config_dict"] = parsed_config_dict
        #process_file(temp_dir, data_results_storage["parsed_config_dict"])
        #data_results_storage["options"] = data_results_storage["parsed_config_dict"]["specname_fitlist"]
    #options = data_results_storage["options"]
    # Optionally, clean up by deleting the temporary files
    # now route to analyse_results
    #return redirect(url_for('generate_synthetic_spectrum'))
    return jsonify({'message': 'Upload successful', 'status': 'success'})

@app.route('/upload_synthetic_spectra', methods=['POST'])
def upload_spectra_synthetic():
    # Get the uploaded files
    file = request.files['file']
    print(file)
    # Temporary directory to save uploaded files
    # create temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        if "/" in file.filename:
            filepath = os.path.join(temp_dir, file.filename.split("/")[1])
        else:
            filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)
        wavelength_observed, flux_observed = np.loadtxt(filepath, unpack=True, usecols=(0,1), dtype=float)
        data_results_storage['observed_spectra_synthetic']["wavelength"] = wavelength_observed
        data_results_storage['observed_spectra_synthetic']["flux"] = flux_observed
        #parsed_config_dict = load_output_data(temp_dir)
        #data_results_storage["parsed_config_dict"] = parsed_config_dict
        #process_file(temp_dir, data_results_storage["parsed_config_dict"])
        #data_results_storage["options"] = data_results_storage["parsed_config_dict"]["specname_fitlist"]
    #options = data_results_storage["options"]
    # Optionally, clean up by deleting the temporary files
    # now route to analyse_results
    #return redirect(url_for('generate_synthetic_spectrum'))
    return jsonify({'message': 'Upload successful', 'status': 'success'})

def process_file(folder_path, processed_dict):
    # linemask loading
    linemask_center_wavelengths, linemask_left_wavelengths, linemask_right_wavelengths = np.loadtxt(processed_dict["linemask_location"], dtype=float, comments=";", usecols=(0, 1, 2), unpack=True)

    # sorts linemask, just like in TSFitPy
    if linemask_center_wavelengths.size > 1:
        linemask_center_wavelengths = np.asarray(sorted(linemask_center_wavelengths))
        linemask_left_wavelengths = np.asarray(sorted(linemask_left_wavelengths))
        linemask_right_wavelengths = np.asarray(sorted(linemask_right_wavelengths))
    elif linemask_center_wavelengths.size == 1:
        linemask_center_wavelengths = np.asarray([linemask_center_wavelengths])
        linemask_left_wavelengths = np.asarray([linemask_left_wavelengths])
        linemask_right_wavelengths = np.asarray([linemask_right_wavelengths])

    data_results_storage["linemask_center_wavelengths"] = linemask_center_wavelengths
    data_results_storage["linemask_left_wavelengths"] = linemask_left_wavelengths
    data_results_storage["linemask_right_wavelengths"] = linemask_right_wavelengths

    output_file_df = processed_dict["output_file_df"]
    fitted_element = processed_dict['fitted_element']

    data_results_storage["fitting_method"] = processed_dict["fitting_method"]

    fitlist = processed_dict["parsed_fitlist"]
    fitlist_parameters: np.ndarray = fitlist.get_spectra_parameters_for_fit(processed_dict["vmic_input_bool"], processed_dict["vmac_input_bool"], processed_dict["rotation_input_bool"])

    if processed_dict["fitting_method"] == "lbl" or processed_dict["fitting_method"] == "vmic":
        if fitted_element != "Fe":
            column_name = f"{fitted_element}_Fe"
            fitted_value_label = f"[{fitted_element}/Fe]"
            data_results_storage["fitted_element"] = fitted_element
        else:
            column_name = "Fe_H"
            fitted_value_label = "[Fe/H]"
            data_results_storage["fitted_element"] = "Fe"
        fitted_value = column_name
    elif processed_dict["fitting_method"] == "teff":
        fitted_value = "Teff"
        fitted_value_label = "Teff"
    elif processed_dict["fitting_method"] == "logg":
        fitted_value = "logg"
        fitted_value_label = "logg"

    data_results_storage["fitted_value_label"] = fitted_value_label


    # load spectra
    for (filename, spectra_rv) in zip(data_results_storage["parsed_config_dict"]["specname_fitlist"], data_results_storage["parsed_config_dict"]["rv_fitlist"]):
        data_results_storage['fitted_spectra'][filename] = {}
        # fitted spectra
        filename_fitted_spectra = os.path.join(folder_path, f"result_spectrum_{filename}_convolved.spec")
        try:
            wavelength, flux = np.loadtxt(filename_fitted_spectra, unpack=True, usecols=(0,1), dtype=float)
            wavelength, flux = zip(*sorted(zip(wavelength, flux)))
        except FileNotFoundError:
            wavelength, flux = [], []
        # argsort wavelength
        wavelength, flux = np.asarray(wavelength), np.asarray(flux)
        data_results_storage['fitted_spectra'][filename]["wavelength_fitted"] = wavelength
        data_results_storage['fitted_spectra'][filename]["flux_fitted"] = flux
        # observed spectra
        wavelength_observed, flux_observed = np.loadtxt(os.path.join(folder_path, filename), unpack=True, usecols=(0,1), dtype=float)
        data_results_storage['fitted_spectra'][filename]["wavelength_observed"] = wavelength_observed
        data_results_storage['fitted_spectra'][filename]["flux_observed"] = flux_observed
        # sensitivity and just blend
        filename_fitted_spectra_just_blend = filename_fitted_spectra.replace("_convolved.spec",
                                                                             "_convolved_just_blend.spec")
        filename_fitted_spectra_minus_sensitivity = filename_fitted_spectra.replace("_convolved.spec",
                                                                                    "_convolved_minus_sensitivity.spec")
        filename_fitted_spectra_plus_sensitivity = filename_fitted_spectra.replace("_convolved.spec",
                                                                                   "_convolved_plus_sensitivity.spec")

        # check if file just blend exists, then load, otherwise set empty arrays
        if os.path.isfile(filename_fitted_spectra_just_blend):
            wavelength_just_blend, flux_just_blend = np.loadtxt(filename_fitted_spectra_just_blend, dtype=float,
                                                                unpack=True)
            # sort the arrays
            indices_sort = np.argsort(wavelength_just_blend)
            wavelength_just_blend = wavelength_just_blend[indices_sort]
            flux_just_blend = flux_just_blend[indices_sort]
        else:
            wavelength_just_blend, flux_just_blend = np.array([]), np.array([])
        wavelength_just_blend, flux_just_blend = list(wavelength_just_blend), list(flux_just_blend)
        # check if file minus sensitivity exists, then load, otherwise set empty arrays
        if os.path.isfile(filename_fitted_spectra_minus_sensitivity):
            wavelength_minus_sensitivity, flux_minus_sensitivity = np.loadtxt(filename_fitted_spectra_minus_sensitivity,
                                                                              dtype=float, unpack=True)
            # sort the arrays
            indices_sort = np.argsort(wavelength_minus_sensitivity)
            wavelength_minus_sensitivity = wavelength_minus_sensitivity[indices_sort]
            flux_minus_sensitivity = flux_minus_sensitivity[indices_sort]
        else:
            wavelength_minus_sensitivity, flux_minus_sensitivity = np.array([]), np.array([])
        wavelength_minus_sensitivity, flux_minus_sensitivity = list(wavelength_minus_sensitivity), list(flux_minus_sensitivity)
        # check if file plus sensitivity exists, then load, otherwise set empty arrays
        if os.path.isfile(filename_fitted_spectra_plus_sensitivity):
            wavelength_plus_sensitivity, flux_plus_sensitivity = np.loadtxt(filename_fitted_spectra_plus_sensitivity,
                                                                            dtype=float, unpack=True)
            # sort the arrays
            indices_sort = np.argsort(wavelength_plus_sensitivity)
            wavelength_plus_sensitivity = wavelength_plus_sensitivity[indices_sort]
            flux_plus_sensitivity = flux_plus_sensitivity[indices_sort]
        else:
            wavelength_plus_sensitivity, flux_plus_sensitivity = np.array([]), np.array([])
        wavelength_plus_sensitivity, flux_plus_sensitivity = list(wavelength_plus_sensitivity), list(flux_plus_sensitivity)
        data_results_storage['fitted_spectra'][filename]["wavelength_just_blend"] = wavelength_just_blend
        data_results_storage['fitted_spectra'][filename]["flux_just_blend"] = flux_just_blend
        data_results_storage['fitted_spectra'][filename]["wavelength_minus_sensitivity"] = wavelength_minus_sensitivity
        data_results_storage['fitted_spectra'][filename]["flux_minus_sensitivity"] = flux_minus_sensitivity
        data_results_storage['fitted_spectra'][filename]["wavelength_plus_sensitivity"] = wavelength_plus_sensitivity
        data_results_storage['fitted_spectra'][filename]["flux_plus_sensitivity"] = flux_plus_sensitivity
        # rv correction
        data_results_storage['fitted_spectra'][filename]['spectra_rv'] = spectra_rv
        # find the spectraname in fitlist_parameters
        index_fitlist = np.where(fitlist_parameters[:, 0] == filename)[0][0]
        # specname, rv_list, teff_list, logg_list, feh_list, vmic_list, vmac_list, rotation_list, abundance_list, resolution_list, snr_list
        data_results_storage['fitted_spectra'][filename]['teff'] = fitlist_parameters[index_fitlist][2]
        data_results_storage['fitted_spectra'][filename]['logg'] = fitlist_parameters[index_fitlist][3]
        #data_results_storage['fitted_spectra'][filename]['feh'] = fitlist_parameters[index_fitlist][4]
        #data_results_storage['fitted_spectra'][filename]['vmic'] = fitlist_parameters[index_fitlist][5]
        #data_results_storage['fitted_spectra'][filename]['vmac'] = fitlist_parameters[index_fitlist][6]
        #data_results_storage['fitted_spectra'][filename]['rotation'] = fitlist_parameters[index_fitlist][7]
        data_results_storage['fitted_spectra'][filename]['abundance_dict'] = fitlist_parameters[index_fitlist][8]
        data_results_storage['fitted_spectra'][filename]['resolution'] = fitlist_parameters[index_fitlist][9]

        if data_results_storage['fitted_spectra'][filename]['resolution'] == 0:
            data_results_storage['fitted_spectra'][filename]['resolution'] = processed_dict["resolution_constant"]

        # load all fitted values for each linemask
        data_results_storage['fitted_spectra'][filename]['fitted_rv'] = {}
        data_results_storage['fitted_spectra'][filename]['flag_error'] = {}
        data_results_storage['fitted_spectra'][filename]['flag_warning'] = {}
        data_results_storage['fitted_spectra'][filename]['chi_squared'] = {}
        data_results_storage['fitted_spectra'][filename]['fitted_value'] = {}
        data_results_storage['fitted_spectra'][filename]['ew'] = {}
        data_results_storage['fitted_spectra'][filename]['vmac'] = {}
        data_results_storage['fitted_spectra'][filename]['rotation'] = {}
        data_results_storage['fitted_spectra'][filename]['vmic'] = {}
        data_results_storage['fitted_spectra'][filename]['Fe_H'] = {}
        data_results_storage['fitted_spectra'][filename]['ew_just_line'] = {}
        data_results_storage['fitted_spectra'][filename]['ew_blend'] = {}
        data_results_storage['fitted_spectra'][filename]['ew_sensitivity'] = {}

        df_correct_specname_indices = output_file_df["specname"] == filename
        output_file_df_specname = output_file_df[df_correct_specname_indices]
        for linemask_idx, linemask_center_wavelength in enumerate(linemask_center_wavelengths):
            #output_file_df_index = output_file_df_specname[output_file_df_specname["wave_center"] == linemask_center_wavelength]
            output_file_df_index = (np.abs(output_file_df_specname[df_correct_specname_indices]["wave_center"] - linemask_center_wavelength)).argmin()
            # load fitted values
            data_results_storage['fitted_spectra'][filename]['fitted_rv'][linemask_idx] = output_file_df_specname["Doppler_Shift_add_to_RV"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['flag_error'][linemask_idx] = output_file_df_specname["flag_error"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['flag_warning'][linemask_idx] = output_file_df_specname["flag_warning"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['chi_squared'][linemask_idx] = output_file_df_specname["chi_squared"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['fitted_value'][linemask_idx] = output_file_df_specname[fitted_value].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['ew'][linemask_idx] = output_file_df_specname["ew"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['vmac'][linemask_idx] = output_file_df_specname["Macroturb"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['rotation'][linemask_idx] = output_file_df_specname["rotation"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['vmic'][linemask_idx] = output_file_df_specname["Microturb"].values[output_file_df_index]
            data_results_storage['fitted_spectra'][filename]['Fe_H'][linemask_idx] = output_file_df_specname["Fe_H"].values[output_file_df_index]
            if "ew_just_line" in output_file_df_specname.columns:
                data_results_storage['fitted_spectra'][filename]['ew_just_line'][linemask_idx] = output_file_df_specname["ew_just_line"].values[output_file_df_index]
            else:
                data_results_storage['fitted_spectra'][filename]['ew_just_line'][linemask_idx] = -999
            if "ew_blend" in output_file_df_specname.columns:
                data_results_storage['fitted_spectra'][filename]['ew_blend'][linemask_idx] = output_file_df_specname["ew_blend"].values[output_file_df_index]
            else:
                data_results_storage['fitted_spectra'][filename]['ew_blend'][linemask_idx] = -999
            if "ew_sensitivity" in output_file_df_specname.columns:
                data_results_storage['fitted_spectra'][filename]['ew_sensitivity'][linemask_idx] = output_file_df_specname["ew_sensitivity"].values[output_file_df_index]
            else:
                data_results_storage['fitted_spectra'][filename]['ew_sensitivity'][linemask_idx] = -999



if __name__ == '__main__':

    app.run(debug=True, port=5002)
    #generate_synthetic_spectrum()
