import sys
import os
import os
import csv
import shutil
import tempfile
import time
import queue
import threading
from pathlib import Path

from io import BytesIO


import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from bsa.segmentation.metrics import iou, dice_coef, dice_loss

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText


import os
from tkinter import *
from tkinter import ttk  

import datetime

import PIL.Image
import PIL.ImageTk

import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import webbrowser

#import webview
from bsa.analysis import custom_indices as custom_index_processing
from bsa.analysis.custom_metric_definitions import CustomMetricDefinition, CustomMetricQueue, CustomMetricDefinitionError
from bsa.analysis.background import AnalysisSnapshot
from bsa.analysis import spectral as spectral_analysis
from bsa.analysis import vegetation_indices
from bsa.exporting import analysis as analysis_export
from bsa.exporting.output_safety import (
	PlannedOutput, OutputCollisionError, publish_staged_outputs, validate_output_plan,
)
from bsa.hyperspectral import dataset as hyperspectral_dataset
from bsa.hyperspectral import hdr as hdr_generation
from bsa.hyperspectral import io as hyperspectral_io
from bsa.hyperspectral import processing as hyperspectral_processing
from bsa.gui.typography import configure_typography
from bsa.gui.hdr_creator import HDRCreatorPanel
from bsa.gui.custom_metrics import CustomMetricEditor
from bsa.gui.analysis_jobs import AnalysisJobController
from bsa.labeling import manifest as label_manifest
from bsa.segmentation import processing as segmentation
from bsa.utils.misc import CustomIndexError, calculate_custom_index, export_results
from bsa.utils.image_input import ImageInputError, decode_raster_image
from bsa.utils.read_paths import require_unchanged, resolve_regular_file
from bsa.utils.resources import ModelFileError, open_in_file_manager, resource_path

application_window = None
menubar = None

answer = None
folderingo = None
RGBCheck = False
ModelCheck = False
importButtonCheck = False
yahu = None
forSingleFile = None
answerLabels = ''

DarkCheck = False
DataCheck = False
BilCheck = False
WhiteCheck = False
MaskCheck = False
useorno = None
#sensorselect = 0
sensorselect = None


class AnalyserInputError(ValueError):
	"""An analyser input needs correction in the GUI."""

#useorno False

def call_analyser_with_dataset_error_dialog(*args, parent=None, **kwargs):
	"""Run the analyser and report dataset or label validation failures."""
	try:
		CallAnalyserExtra(*args, **kwargs)
	except hyperspectral_dataset.DatasetMatchingError as exc:
		messagebox.showerror(
			"Hyperspectral dataset matching failed",
			str(exc),
			parent=parent,
		)
		return False
	except label_manifest.LabelManifestError as exc:
		messagebox.showerror(
			"Label manifest validation failed",
			str(exc),
			parent=parent,
		)
		return False
	except (AnalyserInputError, OutputCollisionError, CustomIndexError, custom_index_processing.CustomIndexNameError, CustomMetricDefinitionError) as exc:
		messagebox.showerror("Analyser input error", str(exc), parent=parent)
		return False
	return True


def inspect_selected_label_manifest(path):
	"""Recognise and structurally validate the selected versioned CSV."""

	return label_manifest.load_label_manifest(path)


def validated_labels_for_dataset(path, dataset, enabled):
	"""Return identity-matched labels, or ``None`` when labels are disabled."""

	if not enabled:
		return None
	return label_manifest.load_and_validate_label_manifest(path, dataset)


def single_dialog_path(selection):
    """Return one selected path, including a sole path wrapped by Tk."""

    if isinstance(selection, (tuple, list)):
        if not selection:
            return None
        if len(selection) != 1:
            raise ValueError("The dialog returned multiple paths; select one folder or file.")
        (selection,) = selection
    if selection is None:
        return None
    try:
        path = os.fsdecode(os.fspath(selection))
    except TypeError as exc:
        raise ValueError("The dialog did not return a file system path.") from exc
    return path or None

def showAbout():
	# Open website 
	#webview.create_window('Sónia Negrão', 'https://people.ucd.ie/sonia.negrao') 
	#webview.start() 
	
	# call webbrowser.open() function. 
	webbrowser.open("https://people.ucd.ie/sonia.negrao") 
	
def restart_program():
	"""Restarts the current program.
	Note: this function does not return. Any cleanup action (like
	saving data) must be done before calling this function."""
	python = sys.executable
	os.execl(python, python, * sys.argv)
	

def close_program():
	application_window.quit()
	application_window.destroy()
	#application_window.quit()

def set_window_icon(window):
    """Set the existing BSA icon with a PNG fallback for non-Windows Tk."""
    try:
        window.iconbitmap(str(resource_path("BSA_logo.ico")))
    except (OSError, tk.TclError):
        try:
            logo_img = PIL.ImageTk.PhotoImage(PIL.Image.open(resource_path("BSA_logo.png")))
            window.iconphoto(True, logo_img)
            window._bsa_icon_image = logo_img
        except (OSError, tk.TclError) as exc:
            if os.environ.get("BSA_DEBUG") == "1":
                print(f"Error loading window icon: {exc}", file=sys.stderr)

def show_about_page():
    """Create a splash screen with structured content."""
    about_window = tk.Toplevel()
    about_window.title("About")
    about_window.geometry("400x500")
    about_window.resizable(False, False)
	
    set_window_icon(about_window)

    # Load the logo
    try:
        logo_image = PIL.Image.open(resource_path("BSA_logo.png"))
        logo_image = logo_image.resize((100, 100), PIL.Image.LANCZOS)
        logo_photo = PIL.ImageTk.PhotoImage(logo_image)
    except Exception as e:
        if os.environ.get("BSA_DEBUG") == "1":
            print(f"Error loading logo: {e}", file=sys.stderr)
        logo_photo = None

    # Load the logo
    try:
        logo_image = PIL.Image.open(resource_path("BSA_logo.png"))
        logo_image = logo_image.resize((100, 100), PIL.Image.LANCZOS)
        logo_photo = PIL.ImageTk.PhotoImage(logo_image)
    except Exception as e:
        if os.environ.get("BSA_DEBUG") == "1":
            print(f"Error loading logo: {e}", file=sys.stderr)
        logo_photo = None

    # Logo Label
    if logo_photo:
        logo_label = tk.Label(about_window, image=logo_photo)
        logo_label.image = logo_photo  # Keep a reference
        logo_label.pack(pady=(20, 5))  # Adjust padding for spacing

    # Title Text
    label_title = ttk.Label(about_window, text="Botanical Spectrum Analyser", font="TkHeadingFont")
    label_title.pack(pady=5)

    # Scrollable Text Widget (Properly formatted paragraph)
    about_text = """Welcome to Botanical Spectrum Analyser - BSA, a user-friendly tool for biologists, breeders, and plant scientists to easily analyse and interpret plant hyperspectral images.

The BSA is a free and open-source software developed for segmenting and analysing hyperspectral imaging. Hyperspectral imaging collects the spectrum for each pixel within a plant image, enabling the detection of stresses such as diseases, abiotic responses, crop monitoring, nutrient levels, etc.

Due to the complexity of hypercubes (3D datasets comprising two spatial dimensions i.e., plant image, and one wavelength dimension), hyperspectral image analysis and segmentation are challenging for non-technical users. 

BSA software is built in Python using Tkinter for the graphical user interface, OpenCV for image processing, and TensorFlow for deep learning operations.

The BSA software enables the accurate segmentation of hyperspectral images and provides real-time analysis of hypercubes. These include an array of pre-calculated or user-specified vegetation indices (e.g., NDVI, NDWI, etc.) and a visualisation of the mean spectra graph of each hypercube."""

    # ScrolledText widget to hold the paragraph text properly
    text_widget = ScrolledText(about_window, wrap=tk.WORD, width=45, height=10, font="TkTextFont")
    text_widget.insert(tk.END, about_text)  # Insert text into the widget
    text_widget.config(state=tk.DISABLED)  # Make it read-only
    text_widget.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    # Close Button
    close_button = ttk.Button(about_window, text="Close", command=about_window.destroy)
    close_button.pack(pady=10)

def maximize_main_window(window):
    """Request one native, decorated maximization before the first Tk draw."""
    methods = (
        (lambda: window.state("zoomed"), lambda: window.attributes("-zoomed", True))
        if sys.platform in ("win32", "darwin") else
        (lambda: window.attributes("-zoomed", True), lambda: window.state("zoomed"))
    )
    for maximize in methods:
        try:
            maximize()
            return
        except (tk.TclError, AttributeError):
            pass
    try:
        width, height = window.winfo_screenwidth(), window.winfo_screenheight()
        if width > 0 and height > 0:
            window.geometry(f"{width}x{height}")
    except (tk.TclError, AttributeError):
        pass


def initialize_application_window():
    """Create the Tk root and authoritative recovered GUI state."""
    global application_window
    global menubar
    global sensorselect
    global useorno
    global yahu

    if application_window is not None:
        return application_window

    application_window = tk.Tk()
    application_window.title('Botanic Spectrum Analyser')
    maximize_main_window(application_window)
    configure_typography(application_window)
    set_window_icon(application_window)

    yahu = tk.StringVar(master=application_window)
    useorno = tk.BooleanVar(master=application_window, value=False)
    sensorselect = tk.IntVar(master=application_window, value=0)

    menubar = Menu(application_window, font="TkMenuFont")
    aboutmenu = Menu(menubar, tearoff=0, font="TkMenuFont")
    aboutmenu.add_command(label="About BSA", command=show_about_page)
    menubar.add_cascade(label="About", menu=aboutmenu)

    helpmenu = Menu(menubar, tearoff=0, font="TkMenuFont")
    helpmenu.add_command(label="Show Video Tutorial", command=lambda: None)
    helpmenu.add_command(label="Show Manual", command=lambda: None)
    menubar.add_cascade(label="Help", menu=helpmenu)

    application_window.config(menu=menubar)
    return application_window

class SegmentationPublicationCancelled(Exception):
	"""Cancellation requested while publishing staged masks."""


class Segmentor:
	def __init__(self, input_dir, output_dir, model_path, progressbar, left):
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.model_path = model_path
		self.progressbar = progressbar
		self.left = left

	def create_dir(self, path):
		"""
		Create a directory if it doesn't exist.
		
		Args:
			path (str): Path of the directory to be created.
		"""
		segmentation.create_directory(path)

	def save_results(self, image, y_pred, save_image_path):
		"""
		Save the segmentation results.
		
		Args:
			image (np.ndarray): Input image.
			y_pred (np.ndarray): Predicted segmentation mask.
			save_image_path (str): Path to save the image.
		"""
		segmentation.save_mask(y_pred, save_image_path)

			
	def pad_image(self, image, new_size=(512, 512)):
		"""Pad an image up to the new size."""
		return segmentation.pad_image(image, new_size)

	def unpad_image(self, image, old_size):
		"""Unpad an image to the old size."""
		return segmentation.unpad_image(image, old_size)

	def resize_image(self, image, new_size=(512, 512)):
		"""Resize an image while maintaining the aspect ratio."""
		return segmentation.resize_image(image, new_size)

						
	def run_segmentation(self, cancelled=None, report=None):
		"""Run segmentation without accessing Tk; report source-image progress."""
		if cancelled is None:
			cancelled = threading.Event()
		if report is None:
			report = lambda event, value: None
		if os.path.lexists(self.output_dir):
			raise OutputCollisionError(
				f"Segmentor run folder already exists: {self.output_dir}."
			)

		test_x = segmentation.discover_image_paths(self.input_dir)
		mask_plan = segmentation.mask_output_plan(test_x, self.output_dir)
		input_files = tuple(
			resolve_regular_file(path, 'RGB image') for path in test_x
		)
		read_paths = tuple(item.path for item in input_files)
		original_plan = tuple(
			PlannedOutput(str(path), "source RGB copy",
				Path(self.output_dir) / 'originals' / Path(path).name)
			for path in test_x
		)
		validate_output_plan((*mask_plan, *original_plan))
		model = segmentation.load_segmentation_model(
			self.model_path,
			lambda path: segmentation.load_keras_model_for_inference(path, tf.keras),
			CustomObjectScope,
			{'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss},
			tf.random.set_seed,
		)
		if cancelled.is_set():
			return False

		report('total', len(test_x))
		Path(self.output_dir).mkdir(parents=True, exist_ok=True)
		with tempfile.TemporaryDirectory(prefix='.bsa-segment-', dir=self.output_dir) as temporary:
			stage = Path(temporary)
			if tf.config.list_physical_devices('GPU'):
				for completed, path in enumerate(test_x, start=1):
					if cancelled.is_set():
						return False
					segmentation.segment_image_file(
						model, path, stage, should_cancel=cancelled.is_set,
						read_path=read_paths[completed - 1],
						expected_file=input_files[completed - 1],
					)
					if cancelled.is_set():
						return False
					report('progress', completed)
			else:
				for completed, (path, _, error) in enumerate(
					segmentation.segment_image_files_cpu(
						model, test_x, stage, should_cancel=cancelled.is_set,
						read_paths=read_paths, expected_files=input_files,
					), start=1
				):
					if cancelled.is_set():
						return False
					if error is not None:
						raise RuntimeError(f"Error processing image {path}: {error}") from error
					report('progress', completed)
			if cancelled.is_set():
				return False
			originals_stage = stage / 'originals'
			originals_stage.mkdir()
			for path, resolved in zip(test_x, input_files):
				require_unchanged(resolved, 'RGB image')
				shutil.copy2(resolved.path, originals_stage / Path(path).name)
				require_unchanged(resolved, 'RGB image')
			(Path(self.output_dir) / 'originals').mkdir(exist_ok=True)
			def check_cancelled():
				if cancelled.is_set():
					raise SegmentationPublicationCancelled()
			try:
				publish_staged_outputs([
					*((item, stage / item.path.name) for item in mask_plan),
					*((item, originals_stage / item.path.name) for item in original_plan),
				], check_cancelled=check_cancelled)
			except SegmentationPublicationCancelled:
				return False
		return True

def CallHDRCreatorExtra(
	samples, lines, bands, bits, byteorder, wavelength_range, file_name, folder, label,
	*, wavelengths=None,
):
	
	
		label.config(text='HDR Creator started...',bg='red')

	
		
		#print ('I AM HERE')
		try:
			output_path = hdr_generation.write_envi_header(
				samples,
				lines,
				bands,
				bits,
				byteorder,
				wavelength_range,
				file_name,
				folder,
				wavelengths=wavelengths,
			)
		except (ValueError, OSError) as error:
			label.config(text='HDR Creator failed. Please check the filename and folder.', bg='red')
			messagebox.showerror('HDR Creator', str(error), parent=application_window)
			return False
		
		label.config(text='HDR Creator finished...Your file is ready at \n '+str(output_path),bg='lightgreen')
		open_in_file_manager(folder)
		return True

	
	
	
#def CallAnalyser(hdrorenvifold,bilfold,marksfol):
def CallAnalyserExtra(hdrdark,hdrdata,hdrwhite,bilfold,marksfold,labelGUI,tab,tree,tc,tabx,button,Resultsframex,resultsShowx,bex1,bex2,bex3,LResFolx,progressbar2,biex3,custom_metrics=None):
	#os.system('python segmentor.py')
	#print(hdrorenvifold)
	global results_df
	global current_results_df
	global outputFileName
	global useorno
	global sensorselect
	global mean_spectra_list
	global wavelengths
	global mean_spectra_dict
	global load
	global setIndex
	
	
	global user_index_name
	global user_formula
	global user_bands

	
	
	labelGUI.config(text='Analyser started...',bg='red')
	progressbar2.pack()
	progressbar2["mode"] = "determinate"
	progressbar2["value"] = 0

	#input_dir = "RGB Images"
	#input_dir = infold
	#output_dir = "Masks"
	#output_dir = outfold
	#model_path = "./Models/Model_UCD_UPJV_VNIR_EPOCH50.h5"
	#model_path = modelselection

	# Create the output directory
	#if not os.path.exists(output_dir):
	#	os.makedirs(output_dir)

	# Image Segmentation
	#segmenter = Segmentor(input_dir, output_dir, model_path)
	#segmenter.run_segmentation()
	


	# Define the size of the hyperspectral cubes
	#cubeSize = [500, 500, 480]

	# Specify the directory path
	#bil_directory_path = 'BIL Files/'
	#bil_directory_path = bilfold
	#hdr_directory_path = 'ENVI Files/'
	#hdr_directory_path = hdrorenvifold
	#mask_directory_path = 'Masks/'
	mask_directory_path = marksfold
	img_directory_path = bilfold
	
	
	# Set the paths for the HDR files (hardcoded)
	#dark_hdr_path = 'HDR Files/Dark_HDR.hdr'
	#data_hdr_path = 'HDR Files/Data_HDR.hdr'
	#white_hdr_path = 'HDR Files/White_HDR.hdr'
	
	dark_hdr_path = hdrdark
	data_hdr_path = hdrdata
	white_hdr_path = hdrwhite
	
	# User specifies the output format (either 'csv' or 'xlsx')
	#output_format = 'xlsx'
	output_format = 'csv'

	# The GUI passes its ordered queue. Keep the direct-call legacy inputs for
	# existing integrations that invoke this callback without the editor.
	if custom_metrics is None:
		metric_definitions = (
			CustomMetricDefinition(
				user_index_name, user_formula,
				user_bands.get("Var1"), user_bands.get("Var2"), user_bands.get("Var3"),
			),
		) if user_index_name else ()
	else:
		metric_definitions = tuple(custom_metrics)
	validation_queue = CustomMetricQueue()
	for metric in metric_definitions:
		validation_queue.add_or_update(metric.name, metric.formula, metric.band_mapping())
	
	
	# Discover and independently sort image/reference/mask files.
	dataset_files = hyperspectral_dataset.discover_dataset(
		img_directory_path,
		mask_directory_path,
	)
	imgFiles = dataset_files.image_files
	maskFiles = dataset_files.mask_files
	dark_img_file_names = dataset_files.dark_files
	data_img_file_names = dataset_files.data_files
	white_img_file_names = dataset_files.white_files
	mask_file_names = dataset_files.mask_files
	numFiles = dataset_files.number_of_files
	numSets = dataset_files.number_of_sets

	# Validate the selected identity manifest before any cube is loaded. The
	# shared validator binds every assignment to the already matched dataset.
	labels_requested = bool(useorno.get() and answerLabels != '')
	validated_labels = validated_labels_for_dataset(
		answerLabels,
		dataset_files,
		labels_requested,
	)
	excluded_unlabelled_count = (
		validated_labels.excluded_unlabelled_count
		if validated_labels is not None
		else 0
	)
	labels_exist = bool(
		validated_labels is not None and validated_labels.labelled_count > 0
	)
	class_names = (
		sorted(set(validated_labels.classes_by_acquisition_id.values()))
		if validated_labels is not None
		else []
	)
	analysis_export.validate_spectrum_identity_names(class_names, numSets)
	# Rebuilt on every run in validated acquisition order. None is retained for
	# ordinary per-sample output but is never treated as a biological class.
	labels = []

	progressbar2["maximum"] = numSets  # ✅ Now numSets is defined

	progress_text_label = ttk.Label(tab, text="Starting analysis...", font="TkSmallCaptionFont")
	progress_text_label.pack()

	start_time = time.time()

	application_window.update()
	application_window.update_idletasks()

	# Create an empty list to store results
	resultsList = []

	# Create a DataFrame from the results list
	results_df = pd.DataFrame(resultsList)

	# Define the filename for the output file (CSV or XLSX)
	#outputFileName = 'Results.csv' 
	
	now = datetime.datetime.now()
		
	# Define the filename for the output file (CSV or XLSX)
	#outputFileName = 'Results' + ('.csv' if output_format == 'csv' else '.xlsx')
	outputFileName = analysis_export.results_filename(now, output_format)
	validate_output_plan((
		PlannedOutput("legacy analysis", "Results report", Path.cwd() / outputFileName),
	))
	# Change the filename and extension as needed

	# Extract the wavelengths being used into memory
	extracted_wavelengths = hyperspectral_io.extract_wavelengths(data_hdr_path)

	# Check that the wavelengths are ok
	if extracted_wavelengths:
		wavelengths = np.array(extracted_wavelengths)  # Convert to numpy array for plotting
	else:
		raise AnalyserInputError("No wavelengths were extracted. Please check the data header file.")

	# Initialize an empty list to store the mean spectra
	mean_spectra_list = []

	# Initialize dictionaries to store mean spectra for each category
	mean_spectra_dict = spectral_analysis.initialize_spectra_by_name(class_names)

	
	combined_stats = {}

	# Built-in PlantScreen modes remain VNIR=0 and SWIR=1; general grid=3.
	sensor_type = 1  # Replace with user input or configuration
	sensor_type = sensorselect.get()
	

	##########################################################################################
	#                                                                                        #
	#                                     Program Run                                        #
	#                                                                                        #
	##########################################################################################

	for setIndex in range(numSets):
	
		progressbar2["value"] = setIndex + 1

		elapsed = time.time() - start_time
		avg_time = elapsed / (setIndex + 1)
		remaining = avg_time * (numSets - setIndex - 1)

		progress_text_label.config(text=f"{setIndex + 1}/{numSets} images processed | ETA: {int(remaining)}s")
		application_window.update_idletasks()
		# Pair files by their independently sorted positions.
		set_paths = hyperspectral_dataset.paths_for_set(dataset_files, setIndex)
		dark_img_path = set_paths.dark
		data_img_path = set_paths.data
		white_img_path = set_paths.white
		mask_path = set_paths.mask

		# Load the three cubes through Spectral Python in the existing order.
		loaded_cubes = hyperspectral_io.load_reference_cubes(
			dark_hdr_path,
			data_hdr_path,
			white_hdr_path,
			dark_img_path,
			data_img_path,
			white_img_path,
		)
		data_arr = loaded_cubes.data
		dark_arr = loaded_cubes.dark
		white_arr = loaded_cubes.white

		# Apply the recovered per-band reference calculation.
		calibrated_data_arr = hyperspectral_processing.calibrate_cube(
			data_arr,
			dark_arr,
			white_arr,
		)

		##########################################################################################
		#                                                                                        #
		#                                       Masking                                          #
		#                                                                                        #
		##########################################################################################

		# Load and apply the stored probability mask without resizing.
		data_masked = hyperspectral_processing.load_and_apply_mask(
			calibrated_data_arr,
			mask_path,
		)
		
		##########################################################################################
		#                                                                                        #
		#                                     Mean Spectrum                                      #
		#                                                                                        #
		##########################################################################################

		# Resolve the class from the acquisition identity, never from row position.
		label = (
			validated_labels.class_for(set_paths)
			if validated_labels is not None
			else None
		)
		labels.append(label)

		mean_spectrum = spectral_analysis.calculate_mean_spectrum(data_masked)
		label = spectral_analysis.record_mean_spectrum_for_class(
			mean_spectra_list,
			mean_spectra_dict,
			mean_spectrum,
			setIndex,
			label,
		)

		# ✅ Store each image separately in dictionary
		image_label = f"Image {setIndex+1}"


		##########################################################################################
		#                                                                                        #
		#                                     Custom Indice                                      #
		#                                                                                        #
		##########################################################################################

		custom_index_stats = {}
		for metric in metric_definitions:
			custom_result = custom_index_processing.calculate_custom_statistics(
				data_masked,
				metric.formula,
				metric.band_mapping(),
				metric.name,
				calculate_custom_index,
			)
			if custom_result.statistics is not None:
				custom_index_stats.update(custom_result.statistics)

		##########################################################################################
		#                                                                                        #
		#                                 Extract file name                                      #
		#                                                                                        #
		##########################################################################################

		# Get the DATA file name without the extension
		dataFileNameWithoutExt = os.path.splitext(data_img_file_names[setIndex])[0]

		# Remove single quotes from the file name using regular expressions
		dataFileNameWithoutExt = re.sub(r"'", '', dataFileNameWithoutExt)

		##########################################################################################
		#                                                                                        #
		#                                 Vegetation Indices                                     #
		#                                                                                        #
		##########################################################################################

		calculated_stats = vegetation_indices.calculate_mode_statistics(
			data_masked,
			sensor_type,
			dataFileNameWithoutExt,
			label,
			custom_index_stats or None,
		)
		if sensor_type == 2:
			stats_to_use = vegetation_indices.update_combined_statistics(
				combined_stats,
				calculated_stats,
			)
		else:
			stats_to_use = calculated_stats
		resultsList.append(stats_to_use)

		# Export results
		current_results_df = pd.DataFrame([stats_to_use])
		results_df = pd.concat([results_df,current_results_df])
		export_results(current_results_df, outputFileName, output_format, setIndex)

	progress_text_label.config(text=f"Done: {numSets}/{numSets} images processed.")


	##########################################################################################
	#                                                                                        #
	#                                    Spectra Viewer                                      #
	#                                                                                        #
	##########################################################################################

	"""# Prepare DataFrame for mean spectra for each BIL file
	mean_spectra_df = pd.DataFrame([wavelengths] + mean_spectra_list).transpose()
	mean_spectra_df.columns = ['Wavelength'] + [f'Mean_Spectrum_{i+1}' for i in range(len(mean_spectra_list))]

	# Export mean spectra DataFrame
	export_spectra_data(mean_spectra_df, 'mean_spectra.' + output_format, output_format)

	if labels_exist:
		# Export mean spectra values per label
		for label in mean_spectra_dict.keys():
			label_spectra_df = pd.DataFrame([wavelengths] + mean_spectra_dict[label]).transpose()
			label_spectra_df.columns = ['Wavelength'] + [f'Mean_Spectrum_{label}_{i+1}' for i in range(len(mean_spectra_dict[label]))]

			# Export label spectra DataFrame
			export_spectra_data(label_spectra_df, f'mean_spectra_{label}.' + output_format, output_format)"""

	# Create the figure and set the size.
	plt.figure(figsize=(10, 6))

	
	
	
	#BU YOLLADIKDAN SONRADIR
	#BU YOLLADIKDAN SONRADIR
	#BU YOLLADIKDAN SONRADIR
	# Plot the mean spectra of each set in gray
	# 1️⃣ **Always plot individual mean spectra per image**
	plt.figure(figsize=(10, 6))
	plot_data = spectral_analysis.prepare_plot_data(
		mean_spectra_dict,
		labels,
		labels_exist,
	)

	if labels_exist:
		# 1️⃣ Plot individual spectra in gray
		for plot_spectrum in plot_data.individual:
			sns.lineplot(x=wavelengths, y=plot_spectrum.values, color='gray', alpha=0.4, linestyle='dotted')

		# 2️⃣ Plot average spectrum per class in color with labels
		for plot_spectrum in plot_data.class_means:
			mean_spectrum = plot_spectrum.values
			sns.lineplot(x=wavelengths, y=mean_spectrum, label=plot_spectrum.label, linewidth=2.5)

		plt.legend(title='Class')  # ✅ Legend only for classes

	else:
		# 3️⃣ No labels file: plot all spectra in unique colors, no legend
		for plot_spectrum in plot_data.individual:
			sns.lineplot(x=wavelengths, y=plot_spectrum.values, linewidth=1)

		# ❌ Explicitly remove any legend if accidentally created
		legend = plt.gca().get_legend()
		if legend:
			legend.remove()

	# Final styling
	plt.xlabel('Wavelength')
	plt.ylabel('Reflectance')
	plt.grid(True)
	plt.tight_layout()


	now = datetime.datetime.now()
		
	# Define the filename for the output file (CSV or XLSX)
	#outputFileName = 'Results-'+now.strftime("%Y%m%d_%H%M%S")+'.csv'  # Change the filename and extension as needed

	'''
	# Export the results DataFrame to CSV or XLSX
	if outputFileName.endswith('.csv'):
		results_df.to_csv('fromOLDVERSION-'+outputFileName, index=False)
	elif outputFileName.endswith('.xlsx'):
		results_df.to_excel('fromOLDVERSION-'+outputFileName, sheet_name='Analysis results', index=False)
	else:
		raise ValueError('Unsupported file format. Use .csv or .xlsx')
	'''	
	completion_status = 'Analyser finished.'
	if validated_labels is not None and excluded_unlabelled_count:
		completion_status += (
			f' {excluded_unlabelled_count} unlabelled acquisition(s) were included '
			'in per-sample analysis and excluded from class-specific analysis.'
		)
	labelGUI.config(text=completion_status,bg='lightgreen')
	progressbar2.stop()
	#label.config(text='Analyser finished.\nYour results are ready in\n'+outputFileName+' and '+outputFileName2+'\n(file located at program folder)',bg='lightgreen')
	
	
	def display_csv_data(file_path,tree):
		
		
		for i in tree.get_children():
			tree.delete(i)
			
		try:
			with open(file_path, 'r', newline='') as file:
				csv_reader = csv.reader(file)
				header = next(csv_reader)  # Read the header row
				tree.delete(*tree.get_children())  # Clear the current data

				tree["columns"] = header
				for col in header:
					tree.heading(col, text=col)
					tree.column(col, width=100)

				for row in csv_reader:
					tree.insert("", "end", values=row)

				#status_label.config(text=f"CSV file loaded: {file_path}")

		except Exception as e:
			#status_label.config(text=f"Error: {str(e)}")
			messagebox.showerror("Results preview failed", str(e), parent=application_window)
	

	
	display_csv_data(outputFileName,tree)
	

	# Create a Scrollbar
	scrollbar = ttk.Scrollbar(tab, orient="horizontal", command=tree.xview)
	scrollbar2 = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)


	
	# Place the scrollbar on the right side of the Treeview
	#scrollbar.pack(side="bottom", fill="x")
	scrollbar.pack(side="top", fill="x")
	scrollbar2.pack(side="right", fill="y")
	
	# Configure the Treeview to use the scrollbar
	tree.configure(xscrollcommand=scrollbar.set)
	tree.configure(yscrollcommand=scrollbar2.set)
	
	tree.pack(padx=1, pady=1, fill="none", expand=False)



	#tree.pack(padx=20, pady=20, fill="both", expand=True)
	#tree.pack(padx=20, pady=20, fill="y", expand=True)
	
	
	button.pack()
	#bex1["state"]="normal"
	#bex2["state"]="normal"
	bex3["state"]="normal"
	biex3["state"]="normal"
	LResFolx.config(text='Please pick a folder to save your results.',bg='white',fg='black')
	#tc.select(tabx)
	
	#with open("test.csv", newline="") as file:
	#with open(outputFileName, newline="") as file:
	#	reader = csv.reader(file)

		# r and c tell us where to grid the labels
	#	for r, col in enumerate(reader):
	#		for c, row in enumerate(col):
				# i've added some styling
	#			labelcsv = tk.Label(
	#				tab, width=10, height=2, text=row, relief=tk.RIDGE
	#			)
	#			labelcsv.grid(row=r, column=c)

	
	
	##########################################################################################
	#                                                                                        #
	#                                    Spectra Viewer                                      #
	#                                                                                        #
	##########################################################################################

	# Save the existing row-oriented image and optional class spectra files.
	analysis_export.write_automatic_spectra(
		wavelengths,
		mean_spectra_dict,
		labels,
		mean_spectra_list,
		labels_exist,
	)
	
	img_data = BytesIO()
	plt.savefig(img_data)

	load = PIL.Image.open(img_data)
	#load = load.resize((500, 300),PIL.Image.LANCZOS) ## The (250, 250) is (height, width)
	load = load.resize((800, 480),PIL.Image.LANCZOS) ## The (250, 250) is (height, width)
	render = PIL.ImageTk.PhotoImage(load)

	#load = Image.open(img_data)
	#render = ImageTk.PhotoImage(load)
	
	#image_ = PIL.Image.open(path)
	#photo = PIL.ImageTk.PhotoImage(n_image)
	
	for widgets in Resultsframex.winfo_children():
		widgets.destroy()

			
	img = tk.Label(Resultsframex, image = render)
	img.photo = render	
	img.image = render # This is needed to keep a reference to the image, see the link below
	img.pack()
	


	#rgb_render = load.convert('RGB')
	#rgb_render.save('C:/myphoto2.jpg', 'JPG')
	
	scrollderoot = tk.Scrollbar(resultsShowx,orient="vertical", command=resultsShowx.yview)
	resultsShowx.configure(yscrollcommand=scrollderoot.set)
	#scrollderoot.pack(side="left",fill="y")
	scrollderoot.pack(side="right",fill="y")
	#scrollderoot.grid(column=5, row=0, sticky='ns', in_=application_window) #instead of number 5, set the column as the expected one for the scrollbar. Sticky ns will might be neccesary.
	
	scrollderootX = tk.Scrollbar(resultsShowx,orient="horizontal", command=resultsShowx.xview)
	resultsShowx.configure(xscrollcommand=scrollderootX.set)
	#scrollderoot.pack(side="left",fill="y")
	scrollderootX.pack(side="bottom",fill="x")
	
	#ysb = tk.Scrollbar(middle, orient="vertical", command=middle.yview)
	#middle.configure(yscrollcommand=ysb.set)
	#ysb.pack(side="right", fill="y")

	
	#xsb = tk.Scrollbar(middle, orient="horizontal", command=middle.xview)
	#middle.configure(xscrollcommand=xsb.set)
	#xsb.pack(side="bottom", fill="x")
		
	#NEW Show the plot
	#plt.show()	


def main():
	initialize_application_window()

	#application_window = tk.Tk()

	# Build a list of tuples for each file type the file dialog should display
	#my_filetypes = [('all files', '.*'), ('text files', '.txt')]

	#os.chdir("..")
	# Ask the user to select a folder.
	#answer = filedialog.askdirectory(parent=application_window,
    #                             initialdir=os.getcwd(),
    #                             title="Please select your input folder (for RGB Images):")

	#os.chdir("..")
	# Ask the user to select a folder.
	#answer2 = filedialog.askdirectory(parent=application_window,
    #                             initialdir=os.getcwd(),
    #                             title="Please select your output folder (for Mask files):")
								 
								 
								 
								 
	#print (len(answer[0]))
	#if len(answer[0])!=1:
	#	for x in range(len(answer)):
	#		print(answer[x])
	#else:
	#		print(answer)
		

	#print (len(answer2[0]))
	#if len(answer2[0])!=1:
	#	for x in range(len(answer2)):
	#		print(answer2[x])
	#else:
	#		print(answer2)

# Ask the user to select a single file name for saving.
#answer = filedialog.asksaveasfilename(parent=application_window,
#                                      initialdir=os.getcwd(),
#                                      title="Please select a file name for saving:",
#                                      filetypes=my_filetypes)

#	B=tk.Button(application_window,text="Start Segmentor",command= CallSegmentation (answer,answer2))

	def selectedCity(self):
		global folderingo
		global RGBCheck
		global ModelCheck
		

			
		selected=optmenu.get()
		
		if selected != "Please Select Model":			
			
			if selected == "IMPORT MY MODEL":
				my_filetypes = [('h5 files','.h5'),('all files', '.*'), ('text files', '.txt')]
				folderingo = filedialog.askopenfile(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select your model:",
                                    filetypes=my_filetypes)
				if folderingo is None:
					ModelCheck=False
					Bstart["state"] = "disabled"
					Bfake = tk.Button(tab3,text="")
					orig_color = Bfake.cget("bg")
					Bstart.configure(bg=orig_color,fg='black')
					l2.config(text='Segmentor\n\nModel NOT Selected. Please select a model from dropdown menu below.',fg='red')
		
				else:
					ModelCheck=True
					l2.config(text='Segmentor\n\n'+selected+' Model Selected. Please load your RGB Image Slices')		
					
					if(RGBCheck):
						Bstart["state"] = "normal"
						#Bsingle["state"] = "normal"
						Bstart.configure(bg="lightgreen")
						#Bstart.configure(bg="lightgreen",fg='white')
						l2.config(text='Segmentor\n\n'+selected+' Model Selected.',fg='green')
			else:
				folderingo = str(resource_path('Models', selected))
				ModelCheck=True
				l2.config(text='Segmentor\n\n'+selected+' Model Selected. Please load your RGB Image Slices')
		
				if(RGBCheck):
					Bstart["state"] = "normal"
					#Bsingle["state"] = "normal"
					Bstart.configure(bg="lightgreen")
					#Bstart.configure(bg="lightgreen",fg='white')
					l2.config(text='Segmentor\n\n'+selected+' Model Selected.',fg='green')
			
		else:
			ModelCheck=False
			Bstart["state"] = "disabled"
			Bfake = tk.Button(tab3,text="")
			orig_color = Bfake.cget("bg")
			Bstart.configure(bg=orig_color,fg='black')
			l2.config(text='Segmentor\n\nModel NOT Selected. Please select a model from dropdown menu below.',fg='red')
		
		
		
		
		
			
		#l2.config(text='Segmentor\n\nModel Selected.')
		return folderingo

	folder = resource_path('Models')
	filelist = [fname[:-3] for fname in os.listdir(folder) if fname.endswith('.h5')] if folder.is_dir() else []



		
	#application_window.geometry('1200x800')
	#application_window.title('Botanic Spectrum Analyser')
	
	tabControl = ttk.Notebook(application_window)
	
	tab1 = ttk.Frame(tabControl)
	tab2 = ttk.Frame(tabControl)
	tab3 = ttk.Frame(tabControl)
	tab4 = ttk.Frame(tabControl)
	tab5 = ttk.Frame(tabControl)

	tabControl.add(tab1, text='Mask Creator')
	tabControl.add(tab2, text='Define Camera Settings')
	tabControl.add(tab3, text='Upload Files')
	tabControl.add(tab4, text='Analyse')
	tabControl.add(tab5, text='Visualise Spectra')
	tabControl.pack(expand=1, fill="both")
	


	#lspace = tk.Label(application_window, bg='white', width=200, text='x',padx=10, pady=10)
	#lspace.pack()
	
	application_window.grid_anchor(anchor='center')
	m = PanedWindow(tab1,orient=HORIZONTAL,sashwidth=10, sashrelief="raised", sashpad=10)
	m.pack(fill="both",expand=1)

	#left = Label(m, text="left pane")
	left = Frame(m,width=200)
	m.paneconfig(left, minsize=200)
	m.add(left)

	#middle = Label(m, text="middle pane")
	#middle = Frame(m,width=200)
	
	#middle = ScrolledText(m, state='disable')
	middle = Text(m, state='disable')
	middle.pack(fill='both', expand=True)
	
	RGBframe = tk.Frame(middle)
	
	middle.window_create('1.0', window=RGBframe)
	m.paneconfig(middle, minsize=200)
	
	#frame = tk.Frame(text)

	m.add(middle)
	
	#scrollbarx = ttk.Scrollbar(RGBframe, orient="horizontal", command=middle.xview)
	#middle.configure(xscrollcommand=scrollbarx.set)
	#scrollbarx.pack(RGBframe,side="bottom", fill="x")
	
	
	xsb = tk.Scrollbar(middle, orient="horizontal", command=middle.xview)
	middle.configure(xscrollcommand=xsb.set)
	xsb.pack(side="bottom", fill="x")
	
	ysb = tk.Scrollbar(middle, orient="vertical", command=middle.yview)
	middle.configure(yscrollcommand=ysb.set)
	ysb.pack(side="right", fill="y")

	
	#right = Label(m, text="right pane")
	right = Text(m, state='disable')
	right.pack(fill='both', expand=True)
	
	Maskframe = tk.Frame(right)
	
	right.window_create('1.0', window=Maskframe)		
	#right = Frame(m,width=200)
	m.paneconfig(right, minsize=200)
	m.add(right)
	
	xsb = tk.Scrollbar(right, orient="horizontal", command=right.xview)
	right.configure(xscrollcommand=xsb.set)
	xsb.pack(side="bottom", fill="x")
	
	ysb = tk.Scrollbar(right, orient="vertical", command=right.yview)
	right.configure(yscrollcommand=ysb.set)
	ysb.pack(side="right", fill="y")
	
	lRGB = tk.Label(middle, bg='white', width=200, text='RGB Image Slices\n(preview of 5)',padx=10, pady=10, justify='center', font="TkHeadingFont")
	lRGB.pack()
	
	lMask = tk.Label(right, bg='white', width=200, text='Mask Images\n(preview of 5)',padx=10, pady=10, justify='center', font="TkHeadingFont")
	lMask.pack()
	
	#l2 = tk.Label(tab1, bg='white', width=200, text='Segmentor\nPlease select which model type you want to use for segmentation:',padx=10, pady=10, justify='left')
	l2 = tk.Label(left, bg='white', width=200, text='Segmentor\nSelect Model:',padx=10, pady=10, justify='center', font="TkHeadingFont")
	#l2.grid(row=1,column=0)
	l2.pack()
	
	#progressbar = ttk.Progressbar(mode="indeterminate")
	progressbar = ttk.Progressbar(left)
	
	segmentation_job = None
	segmentation_closing = False
	analysis_controller = AnalysisJobController(application_window)
	analysis_control_states = []
	analysis_table_scrollbars = []
	analysis_preview_scrollbars = []

	def show_finished_masks(input_dir, output_dir):
		"""Restore the existing output-folder and mask-preview workflow on Tk."""
		open_in_file_manager(output_dir[2:])
		Bsingle["state"] = "normal"
		for widget in Maskframe.winfo_children():
			widget.destroy()
		for _ in range(3):
			tk.Label(Maskframe, text='').pack()
		count = 0
		for file_ in sorted(os.listdir(output_dir)):
			if not file_.endswith('-PlantMask.png'):
				continue
			count += 1
			if count < 6:
				photo_name = tk.Label(Maskframe, text='IMAGE #' + str(count) + ': ' + file_)
				path = os.path.join(output_dir, file_)
				image_ = decode_raster_image(path, mask=True)
				n_image = image_.resize((500, 500))
				photo = PIL.ImageTk.PhotoImage(n_image)
				photo_name.pack()
				img_label = tk.Label(Maskframe, image=photo)
				img_label.photo = photo
				img_label.pack()

	def finish_segmentation_job(job, outcome, error=None):
		nonlocal segmentation_job
		if segmentation_closing or segmentation_job is not job:
			return
		segmentation_job = None
		if job['total'] == 0:
			progressbar.stop()
		for widget, state in job['control_states']:
			widget.configure(state=state)
		if outcome == 'cancelled':
			job['status'].config(text='Segmentor stopped.', bg='red')
			return
		if outcome == 'error':
			job['status'].config(text='Segmentor stopped. Please check the error.', bg='red')
			title = 'Model Error' if isinstance(error, ModelFileError) else 'Segmentation Error'
			messagebox.showerror(title, str(error), parent=application_window)
			return
		job['status'].config(
			text='Segmentor finished.\nFiles are ready in ' + job['output_dir'][2:]
			+ ' folder.\nPlease select which model type you want to (re)start the segmentation:',
			bg='lightgreen',
		)
		job['progress_label'].config(
			text=f"Done: {job['total']}/{job['total']} images segmented."
		)
		try:
			show_finished_masks(job['input_dir'], job['output_dir'])
		except Exception as exc:
			messagebox.showerror('Segmentation Error', str(exc), parent=application_window)
			return
		messagebox.showinfo('Segmentation complete', 'Segmentation finished.', parent=application_window)

	def poll_segmentation_job(job):
		if segmentation_closing or segmentation_job is not job:
			return
		while True:
			try:
				event, value = job['events'].get_nowait()
			except queue.Empty:
				break
			if event == 'total':
				job['total'] = value
				progressbar.stop()
				progressbar.configure(mode='determinate', maximum=value, value=0)
			elif event == 'progress':
				progressbar['value'] = value
				elapsed = time.monotonic() - job['start_time']
				remaining = elapsed / value * (job['total'] - value)
				job['progress_label'].config(
					text=f"{value}/{job['total']} images segmented | ETA: {int(remaining)}s"
				)
			elif event in ('done', 'cancelled', 'error'):
				finish_segmentation_job(job, event, value)
				return
		job['after_id'] = application_window.after(30, lambda: poll_segmentation_job(job))

	def CallSegmentationExtra(infold, outfold, modelselection, label):
		global forSingleFile
		nonlocal segmentation_job
		if segmentation_closing or segmentation_job is not None:
			return
		input_dir = infold
		output_dir = (outfold if getattr(sys, 'frozen', False)
		              else segmentation.timestamped_output_directory(datetime.datetime.now()))
		forSingleFile = output_dir
		progressbar.pack()
		progressbar.configure(mode='indeterminate')
		progressbar.start()
		label.config(text='Segmentor started...', bg='red')
		progress_label = ttk.Label(left, text='Starting segmentation...', font='TkSmallCaptionFont')
		progress_label.pack()
		controls = (Bstart, Bfolder, optmenu, btton1)
		control_states = [(widget, widget.cget('state')) for widget in controls]
		for widget in controls:
			widget.configure(state='disabled')
		job = {
			'events': queue.Queue(),
			'cancelled': threading.Event(),
			'control_states': control_states,
			'status': label,
			'progress_label': progress_label,
			'input_dir': input_dir,
			'output_dir': output_dir,
			'start_time': time.monotonic(),
			'total': 0,
			'after_id': None,
		}
		segmentation_job = job
		segmenter = Segmentor(input_dir, output_dir, modelselection, progressbar, left)

		def worker():
			try:
				finished = segmenter.run_segmentation(
					job['cancelled'], lambda event, value: job['events'].put((event, value))
				)
				job['events'].put(('done' if finished else 'cancelled', None))
			except Exception as exc:
				job['events'].put(('error', exc))

		job['thread'] = threading.Thread(target=worker, name='BSA segmentation', daemon=True)
		job['thread'].start()
		job['after_id'] = application_window.after(30, lambda: poll_segmentation_job(job))

	def CallSegmentationInit():

		CallSegmentationExtra2()
		
	def CallSegmentationExtra2():
		
		yarro = str(resource_path('Models', optmenu.get()+'.h5'))
		if optmenu.get() != 'IMPORT MY MODEL' and optmenu.get() != 'Please Select Model':
			yarro = folderingo+'.h5'
		elif importButtonCheck==True:
			yarro = folderingo.name
		else:	
			yarro = folderingo.name
			
		
		
		CallSegmentationExtra(answer,answer2,yarro,l2)
	
	
	#optmenu = ttk.Combobox(tab1, values=filelist, state='readonly',justify='center')
	
	l = list(filelist)
	l.insert(0,'Please Select Model')
	l.insert(len(filelist)+1,'IMPORT MY MODEL')
	filelist3=tuple(l)
	#for x in range(len(l)-1):
	#	print(filelist)
	#	print(filelist2)
		#filelist2 += (str(filelist[x]))
	#	l.insert(x+1,filelist[x])
		
		
	optmenu = ttk.Combobox(left, values=filelist3, state='readonly',justify='center')
	#if len(filelist)==0:
	#	#optmenu['values'] += ('IMPORT MY MODEL',)
	#	optmenu['values'] = (*optmenu['values'], 'Please Select Model')
	#	optmenu['values'] = (*optmenu['values'], 'IMPORT MY MODEL')
	#else:

	#	optmenu['values'] += ('IMPORT MY MODEL',)
	#optmenu.pack(fill='x')
	#optmenu.grid(row=2,column=0)
	optmenu.pack(fill='none')
	
	optmenu.current(0)
	optmenu.bind("<<ComboboxSelected>>",selectedCity)
	
	
	
	#l2 = tk.Label(application_window, bg='white', width=200, text='Segmentor\nPlease select which model type you want to use for segmentation:',padx=10, pady=10)
	#l2.pack()
	
	#optmenu = ttk.Combobox(application_window, values=filelist, state='readonly',justify='center')
	#optmenu.pack(fill='x')
	#optmenu.current(0)
	#optmenu.bind("<<ComboboxSelected>>",selectedCity)
	
	#yarro = './Models/'+str(selectedCity)

	


	
	
	
	def CallConventor():
		lconv.config(text='I am tring to run...')
		
	def CallHDRCreator():

		lhdr.config(text='HDR Creator\nRunning (asking for input)...',bg='red')
		l.config(bg='white')
		l2.config(bg='white')
		#B["state"] = "disable"
		#B2["state"] = "disable"
		
		
		#B["state"] = "normal"
		#B2["state"] = "normal"
		


		samples = simpledialog.askinteger("Samples:", "Enter the number of samples:",
											 parent=application_window,
											 minvalue=1, maxvalue=hdr_generation.MAX_CREATOR_DIMENSION)
			
		lines = simpledialog.askinteger("lines", "Enter the number of lines:",
											 parent=application_window,
											 minvalue=1, maxvalue=hdr_generation.MAX_CREATOR_DIMENSION)
			
		bands = simpledialog.askinteger("Bands:", "Enter the number of bands:",
											 parent=application_window,
											 minvalue=2, maxvalue=hdr_generation.MAX_CREATOR_BANDS)
			
		bits = simpledialog.askinteger("ENVI Data Type Code:", "Enter a supported ENVI data type code:",
											 parent=application_window,
											 minvalue=0, maxvalue=99)
			
		byteorder = simpledialog.askinteger("Byte Order:", "Enter the byte order: ",
											 parent=application_window,
											 minvalue=0, maxvalue=1)

		starting_wavelength = simpledialog.askstring("Starting Wavelength:", "Enter the starting wavelength:",
									   parent=application_window)
			
		ending_wavelength = simpledialog.askstring("Ending Wavelength:", "Enter the ending wavelength: ",
									   parent=application_window)
			
			
		file_name = simpledialog.askstring("File Name:", "Enter the file name:",
                                parent=application_window)
		
	
    
    

		lhdr.config(text='HDR Creator\nRunning (creating files based on input)...',bg='red')
		print ('HDR Creator started.')
		if any(value is None for value in (
			samples, lines, bands, bits, byteorder, starting_wavelength,
			ending_wavelength, file_name,
		)):
			lhdr.config(text='HDR Creator cancelled.', bg='white')
			return False
		folder = globals().get('HDRF', '')
		if not folder:
			messagebox.showerror(
				'HDR Creator', 'Select an output folder before creating the header.',
				parent=application_window,
			)
			return False
		return CallHDRCreatorExtra(
			samples, lines, bands, bits, byteorder,
			(starting_wavelength, ending_wavelength), file_name, folder, lhdr,
		)
		


	def CallSegmentation():
		nonlocal answer2
		global answer
		
		global RGBCheck
		global ModelCheck
		

		selected = filedialog.askdirectory(parent=application_window,
									 initialdir=os.getcwd(),
									 title="Please select your input folder (for RGB Images):")
		try:
			selected_directory = single_dialog_path(selected)
		except ValueError as exc:
			messagebox.showerror("RGB Slices folder", str(exc), parent=application_window)
			return
		if selected_directory is None:
			return
		if not os.path.isdir(selected_directory):
			messagebox.showerror(
				"RGB Slices folder", "Please select an existing RGB Slices folder.",
				parent=application_window,
			)
			return
		output_parent = None
		if getattr(sys, 'frozen', False):
			output_parent = filedialog.askdirectory(
				parent=application_window,
				initialdir=str(Path.home()),
				title='Please select the folder for generated masks:',
			)
			if not output_parent:
				return
		answer = selected_directory
		l2.config(text='RGB Image Slices are ready...',)
		l.config(bg='white')

		#os.chdir("..")
		# Ask the user to select a folder.
		
		#THIS IS EDITED FOR THE FLOW OF THE APP
		
		#answer2 = filedialog.askdirectory(parent=application_window,
		#							 initialdir=os.getcwd(),
		#							 title="Please select your output folder (for Mask files):")
		#answer2 = 'no'
		now = datetime.datetime.now()
		mask_folder_name = 'Masks-'+now.strftime("%Y%m%d_%H%M%S")
		answer2 = str(Path(output_parent) / mask_folder_name) if output_parent else './'+mask_folder_name
		
		#print ('./Models/'+optmenu.get())
		
		yarro = str(resource_path('Models', optmenu.get()+'.h5'))
		
		
		for widgets in RGBframe.winfo_children():
			widgets.destroy()

	
		spaceforImagesMiddle = tk.Label(RGBframe,text='')
		spaceforImagesMiddle.pack()
		spaceforImagesMiddle2 = tk.Label(RGBframe,text='')
		spaceforImagesMiddle2.pack()
		spaceforImagesMiddle3 = tk.Label(RGBframe,text='')
		spaceforImagesMiddle3.pack()

		count=0
		for (root_, dirs, files) in os.walk(answer):
			if files:
				for file_ in files:
					if(file_[-3:]!='png'):
						pass
					else:
						count=count+1
						if count>5:
							break
						photo_name = tk.Label(RGBframe,text='IMAGE #'+str(count)+': '+file_)
						path = os.path.join(answer, file_)
						try:
							image_ = decode_raster_image(path)
						except ImageInputError as exc:
							messagebox.showerror('RGB Image Preview', str(exc), parent=application_window)
							return
						n_image = image_.resize((500, 500))
						photo = PIL.ImageTk.PhotoImage(n_image)
						photo_name.pack()
						img_label = tk.Label(RGBframe, image=photo)
						img_label.photo = photo                             # <--
						img_label.pack()
						#break
							

		'''	
		references = []
		for images in os.listdir(answer):
			if images.endswith("png"):
				im = PIL.Image.open(os.path.join(answer,images))
				tkimage = PIL.ImageTk.PhotoImage(im)
				references.append(tkimage) 

		for image in references:
		    Label(tab1,image=image).pack()

			#img_label = tk.Label(tab1, image=tkimage)
			#img_label.pack()				
		
				#handler = lambda img = images: getFileName(img)  #here modify
				#imageButton = Button(tab1, image=tkimage, command=handler)#here
				#imageButton.image=tkimage
				#imageButton.pack()
		'''
		RGBCheck=True
		
		
		if(ModelCheck and RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
		#CallSegmentationExtra(answer,answer2,yarro,l2)
		#os.startfile(answer2[2:])
		
		#l.config(text='Segmentor finished.')
		#B2["state"] = "normal"
		#B4["state"] = "normal"
		#os.system(answer2)


	def ExportXLSX():
		global results_df
		global current_results_df
		global setIndex
		
		now = datetime.datetime.now()
			
		# Define the filename for the output file (CSV or XLSX)
		outputFileName2 = analysis_export.results_filename(now, 'xlsx')  # Change the filename and extension as needed
		
		'''		
		# Export the results DataFrame to CSV or XLSX
		if outputFileName2.endswith('.csv'):
			results_df.to_csv(ResFol+'/'+outputFileName2, index=False)
		elif outputFileName2.endswith('.xlsx'):
			results_df.to_excel(ResFol+'/'+outputFileName2, sheet_name='Analysis results', index=False)
		else:
			raise ValueError('Unsupported file format. Use .csv or .xlsx')

		l.config(text='Your results are ready in\n'+outputFileName2+'\n at '+ResFol,bg='lightgreen')
		open_in_file_manager(ResFol)
		#os.startfile(os.getcwd())
		#os.startfile(os.getcwd()+'/'+outputFileName2)
		'''
		#export_results(current_results_df, ResFol+'/'+outputFileName2, 'xlsx', 0)
		try:
			export_results(results_df, os.path.join(ResFol, outputFileName2), 'xlsx', 0)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
		
	def ExportCSV():
		global outputFileName
		global current_results_df
		global setIndex
		'''
		# Export the results DataFrame to CSV or XLSX
		if outputFileName.endswith('.csv'):
			results_df.to_csv(ResFol+'/'+outputFileName, index=False)
		elif outputFileName.endswith('.xlsx'):
			results_df.to_excel(ResFol+'/'+outputFileName, sheet_name='Analysis results', index=False)
		else:
			raise ValueError('Unsupported file format. Use .csv or .xlsx')		
		
		
		#ResFol+'/'+
		l.config(text='Your results are ready in\n'+outputFileName+'\n at '+ResFol,bg='lightgreen')
		open_in_file_manager(ResFol)
		#os.startfile(os.getcwd())
		#os.startfile(os.getcwd()+'/'+outputFileName)
		'''
		#export_results(current_results_df, ResFol+'/'+outputFileName, 'csv', 0)
		try:
			export_results(results_df, os.path.join(ResFol, outputFileName), 'csv', 0)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
	
	def BrowseForExport():
		global ResFol
		ResFol = filedialog.askdirectory(parent=application_window,
					initialdir=os.getcwd(),
					title="Please select the folder to save Results:")
					
		if ResFol != '':
			LResFol.config(text='Folder: '+ResFol,bg='white',fg='black')
			Bexport1["state"]="normal"
			Bexport2["state"]="normal"
		else:
			LResFol.config(text='Please pick a folder to save your results.',bg='white',fg='black')
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"

	
	spectra_output_directory = ''
	spectra_labels_exist = False
	current_plot_data = None
	current_plot_grid = None
	class_colours = {}
	class_colour_buttons = {}

	def _choose_spectra_output_directory():
		nonlocal spectra_output_directory
		selected = filedialog.askdirectory(
			parent=application_window,
			initialdir=spectra_output_directory or os.getcwd(),
			title="Please select the folder to save Image and Mean Spectra files:",
		)
		if selected:
			spectra_output_directory = selected
			LResFol2.config(text='Folder: '+selected, bg='white', fg='black')
		return selected or None

	def BrowseForExportImg():
		_choose_spectra_output_directory()

	def _spectra_export_directory():
		return spectra_output_directory or _choose_spectra_output_directory()

	def _set_visualisation_ready(ready):
		state = 'normal' if ready else 'disabled'
		BIexport1.configure(state=state)
		BIexportImg.configure(state=state)

	def _replace_visualisation():
		nonlocal spectra_labels_exist, current_plot_data, current_plot_grid
		global load, mean_spectra_list, mean_spectra_dict, wavelengths
		previous = globals().get('load')
		if previous is not None:
			previous.close()
		load = None
		mean_spectra_list = []
		mean_spectra_dict = spectral_analysis.initialize_spectra_by_name(())
		spectra_labels_exist = False
		current_plot_data = None
		current_plot_grid = None
		class_colours.clear()
		class_colour_buttons.clear()
		for widget in class_colours_frame.winfo_children():
			widget.destroy()
		class_colours_frame.pack_forget()
		wavelengths = np.array([])
		for widget in Resultsframe.winfo_children():
			widget.destroy()
		_set_visualisation_ready(False)
	
	def ExportCSVMean():
		global mean_spectra_list
		global wavelengths
		global mean_spectra_dict
		folder = _spectra_export_directory()
		if folder is None:
			return
			# Export the mean spectra values for each BIL file
			
		now = datetime.datetime.now()
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

		
		try:
			analysis_export.write_timestamped_mean_spectra(
				folder, wavelengths, mean_spectra_list, mean_spectra_dict,
				spectra_labels_exist, now, 'csv',
			)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
			return
		open_in_file_manager(folder)
		
	def ExportXLSXMean():
		global mean_spectra_list
		global wavelengths
		folder = _spectra_export_directory()
		if folder is None:
			return

		
			# Export the mean spectra values for each BIL file
			
		now = datetime.datetime.now()
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

		
		try:
			analysis_export.write_timestamped_mean_spectra(
				folder, wavelengths, mean_spectra_list, mean_spectra_dict,
				spectra_labels_exist, now, 'xlsx',
			)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
			return
		open_in_file_manager(folder)
	
	def ExportIMG():
		global load
		folder = _spectra_export_directory()
		if folder is None:
			return
		
		now = datetime.datetime.now()
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
		
		try:
			analysis_export.save_plot_pair(load, folder, now)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
			return
		open_in_file_manager(folder)

	def ExportIMGJPG():
		global load
		folder = _spectra_export_directory()
		if folder is None:
			return
		
		now = datetime.datetime.now()
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
		
		try:
			analysis_export.save_jpeg(load, folder, now)
		except (OutputCollisionError, OSError) as error:
			messagebox.showerror('Export error', str(error), parent=application_window)
			return
		open_in_file_manager(folder)
		
	def JumpToA():
		tabControl.select(tab4)	
	
	def JumpToU():
		tabControl.select(tab3)	

	def JumpToV():
		tabControl.select(tab5)			
		
	
	def _set_upload_message(message):
		lheader3.config(text=message)
		if message:
			lheader3.grid(row=1, column=0, columnspan=2, sticky='w', padx=6, pady=(8, 2))
		else:
			lheader3.grid_remove()

	def _set_labels_message(message):
		label_status.config(text=message)
		if message:
			label_status.pack(anchor='w', padx=8, pady=(6, 2))
		else:
			label_status.pack_forget()

	def Dark():
		global answer3
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('hdr files', '.hdr'),('all files', '.*'), ('text files', '.txt')]

		# Ask the user to select a single file name.
		previous_header = globals().get("answer3")
		selected = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a Dark HDR file:",
                                    filetypes=my_filetypes)
		if not selected:
			return
		answer3 = selected
		if answer3 != previous_header:
			metric_editor.other_header_changed()
		#print('yow'+answer3)
		Bfake = tk.Button(tab3,text="")
		orig_color = Bfake.cget("bg")
		if answer3 !='':
		#	print (answer3)
			B2dark.config(bg="lightgreen")
			DarkCheck=True

			#print (DarkCheck)
			#print (DataCheck)
			#print (WhiteCheck)
			#print (BilCheck)
			#print (MaskCheck)
			
			if (DarkCheck and DataCheck and WhiteCheck and BilCheck and MaskCheck):
				
				B2["state"] = "normal"
				B2.config(bg='lightgreen')
				B2jump.pack(side = BOTTOM)
		else:
			B2dark.config(bg=orig_color)
			DarkCheck=False
			B2["state"] = "disabled"
			B2.config(bg=orig_color)
			B2jump.forget()
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"
			Bexport3["state"] = "disabled"
			
	def Data():
		global answer3b
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('hdr files', '.hdr'),('all files', '.*'), ('text files', '.txt')]
		# Ask the user to select a single file name.
		selected = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a Data HDR file:",
                                    filetypes=my_filetypes)
		if not selected:
			return
		answer3b = selected
		metric_editor.data_header_changed(answer3b or None)
		
		Bfake = tk.Button(tab3,text="")
		orig_color = Bfake.cget("bg")
		if answer3b !='':
			B2data.config(bg="lightgreen")
			DataCheck=True
			
			if (DarkCheck and DataCheck and WhiteCheck and BilCheck and MaskCheck):
				B2["state"] = "normal"
				B2.config(bg='lightgreen')
				B2jump.pack(side = BOTTOM)
			
		else:
			B2data.config(bg=orig_color)
			DataCheck=False
			B2["state"] = "disabled"
			B2.config(bg=orig_color)
			B2jump.forget()
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"
			Bexport3["state"] = "disabled"
			
		
	
	def White():
		global answer3c
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('hdr files', '.hdr'),('all files', '.*'), ('text files', '.txt')]
		# Ask the user to select a single file name.
		previous_header = globals().get("answer3c")
		selected = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a White HDR file:",
                                    filetypes=my_filetypes)
		if not selected:
			return
		answer3c = selected
		if answer3c != previous_header:
			metric_editor.other_header_changed()
		Bfake = tk.Button(tab3,text="")
		orig_color = Bfake.cget("bg")
		if answer3c !='':
			B2white.config(bg="lightgreen")
			WhiteCheck=True

			if (DarkCheck and DataCheck and WhiteCheck and BilCheck and MaskCheck):
				B2["state"] = "normal"
				B2.config(bg='lightgreen')
				B2jump.pack(side = BOTTOM)
		else:
			B2white.config(bg=orig_color)
			WhiteCheck=False
			B2["state"] = "disabled"
			B2.config(bg=orig_color)
			B2jump.forget()
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"
			Bexport3["state"] = "disabled"
			
	def Bil():
		global answer4
		global answer5
		global answerLabels
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('BIL files', '.bil'),('RAW files', '.raw'),('all files', '.*'), ('text files', '.txt')]
		selected = filedialog.askdirectory(parent=application_window,
                                 initialdir=os.getcwd(),
                                 title="Please select your input folder (for BIL files):")
		if not selected:
			return
		if selected != globals().get('answer4'):
			previous_folder = globals().get('answer4')
			answer5 = ''
			MaskCheck = False
			B2mask.configure(bg=B2mask_default_bg)
			answerLabels = ''
			useorno.set(False)
			_set_labels_message('')
			if previous_folder:
				_set_upload_message('Input folder changed. Select matching masks and, if needed, Labels again.')
			B2['state'] = 'disabled'
			B2jump.forget()
		answer4 = selected
		Bfake = tk.Button(tab3,text="")
		orig_color = Bfake.cget("bg")
		if answer4 !='':
			B2bil.config(bg="lightgreen")
			BilCheck=True
			if (DarkCheck and DataCheck and WhiteCheck and BilCheck and MaskCheck):
				B2["state"] = "normal"
				B2.config(bg='lightgreen')
				B2jump.pack(side = BOTTOM)

		else:
			B2bil.config(bg=orig_color)
			BilCheck=False
			B2["state"] = "disabled"
			B2.config(bg=orig_color)
			B2jump.forget()
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"
			Bexport3["state"] = "disabled"
			
	def Mask():
		global answer5
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck

		my_filetypes = [('PNG files', '.png'),('all files', '.*'), ('text files', '.txt')]
		selected = filedialog.askdirectory(parent=application_window,
									 initialdir=os.getcwd(),
									 title="Please select your Masks folder (Segmentor output folder):")
		if not selected:
			return
		answer5 = selected
		_set_upload_message('')
	
		Bfake = tk.Button(tab3,text="")
		orig_color = Bfake.cget("bg")
		if answer5 !='':
			B2mask.config(bg="lightgreen")
			MaskCheck=True
			if (DarkCheck and DataCheck and WhiteCheck and BilCheck and MaskCheck):
				B2["state"] = "normal"
				B2.config(bg='lightgreen')
				B2jump.pack(side = BOTTOM)

		else:
			B2mask.config(bg=orig_color)
			MaskCheck=False
			B2["state"] = "disabled"
			B2.config(bg=orig_color)
			B2jump.forget()
			Bexport1["state"] = "disabled"
			Bexport2["state"] = "disabled"
			Bexport3["state"] = "disabled"
			
	def _analysis_input_controls():
	    controls = [B2, btton1, Bexport1, Bexport2, Bexport3, BIexport3]
	    def visit(parent):
	        for child in parent.winfo_children():
	            if child is not B2jump and isinstance(child, (tk.Button, ttk.Button, tk.Radiobutton, tk.Checkbutton, tk.Entry, ttk.Entry, ttk.Combobox)):
	                controls.append(child)
	            visit(child)
	    visit(tab3)
	    return controls

	def _set_analysis_busy(busy):
	    nonlocal analysis_control_states
	    if busy:
	        analysis_control_states = [(widget, str(widget.cget('state'))) for widget in _analysis_input_controls()]
	        for widget, _state in analysis_control_states:
	            widget.configure(state='disabled')
	    else:
	        for widget, state in analysis_control_states:
	            if widget.winfo_exists():
	                widget.configure(state=state)
	        analysis_control_states = []
	    metric_editor.set_busy(busy)

	def _validate_visualisation(grid, plot_data):
	    values = np.asarray(grid, dtype=float)
	    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
	        raise ValueError('Spectra visualisation requires finite wavelengths.')
	    curves = (*plot_data.individual, *plot_data.class_means)
	    if not curves:
	        raise ValueError('No spectra are available to visualise.')
	    has_visible_value = False
	    for curve in curves:
	        spectrum = np.asarray(curve.values, dtype=float)
	        if spectrum.ndim != 1 or spectrum.size != values.size:
	            raise ValueError('A spectrum does not match the wavelength count.')
	        has_visible_value |= bool(np.isfinite(spectrum).any())
	    if not has_visible_value:
	        raise ValueError('No finite spectral values are available to visualise.')

	def _render_spectra_image(grid, plot_data, selected_colours=None):
	    figure = plt.figure(figsize=(10, 6))
	    plotted_colours = {}
	    try:
	        if plot_data.labelled:
	            for item in plot_data.individual:
	                sns.lineplot(x=grid, y=item.values, color='gray', alpha=0.4, linestyle='dotted')
	            for item in plot_data.class_means:
	                options = {'color': selected_colours[item.label]} if selected_colours is not None else {}
	                sns.lineplot(x=grid, y=item.values, label=item.label, linewidth=2.5, **options)
	                plotted_colours[item.label] = to_hex(plt.gca().lines[-1].get_color())
	            plt.legend(title='Class')
	        else:
	            for item in plot_data.individual:
	                sns.lineplot(x=grid, y=item.values, linewidth=1)
	            legend = plt.gca().get_legend()
	            if legend:
	                legend.remove()
	        plt.xlabel('Wavelength')
	        plt.ylabel('Reflectance')
	        plt.grid(True)
	        plt.tight_layout()
	        image_bytes = BytesIO()
	        figure.savefig(image_bytes)
	        image_bytes.seek(0)
	        with PIL.Image.open(image_bytes) as image:
	            rendered = image.resize((800, 480), PIL.Image.LANCZOS)
	        return rendered, plotted_colours
	    finally:
	        plt.close(figure)

	def _display_spectra_image(image):
	    global load
	    render = PIL.ImageTk.PhotoImage(image)
	    for widget in Resultsframe.winfo_children():
	        widget.destroy()
	    preview = tk.Label(Resultsframe, image=render)
	    preview.photo = render
	    preview.pack()
	    previous = globals().get('load')
	    load = image
	    if previous is not None:
	        previous.close()

	def _choose_class_colour(label):
	    proposed = colorchooser.askcolor(
	        color=class_colours[label], title=f'Choose colour for {label}', parent=application_window,
	    )
	    if not proposed or not proposed[1]:
	        return
	    try:
	        colour = to_hex(proposed[1])
	    except ValueError:
	        return
	    if colour == class_colours[label]:
	        return
	    updated = {**class_colours, label: colour}
	    image, _ = _render_spectra_image(current_plot_grid, current_plot_data, updated)
	    _display_spectra_image(image)
	    class_colours.update(updated)
	    class_colour_buttons[label].configure(fg=colour, activeforeground=colour)

	def _show_class_colour_controls(plot_data, initial_colours):
	    class_colours.update(initial_colours)
	    if not plot_data.labelled or not plot_data.class_means:
	        return
	    canvas = tk.Canvas(class_colours_frame, height=min(128, 32 * len(plot_data.class_means)),
	                       width=360, highlightthickness=0)
	    canvas.pack(side='left', fill='x', expand=True)
	    rows = ttk.Frame(canvas)
	    window = canvas.create_window((0, 0), window=rows, anchor='nw')
	    rows.bind('<Configure>', lambda _event: canvas.configure(scrollregion=canvas.bbox('all')))
	    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
	    if len(plot_data.class_means) > 4:
	        scrollbar = ttk.Scrollbar(class_colours_frame, orient='vertical', command=canvas.yview)
	        scrollbar.pack(side='right', fill='y')
	        canvas.configure(yscrollcommand=scrollbar.set)
	    for item in plot_data.class_means:
	        row = ttk.Frame(rows)
	        row.pack(fill='x')
	        ttk.Label(row, text=item.label).pack(side='left', fill='x', expand=True)
	        colour = class_colours[item.label]
	        button = tk.Button(row, text='■', fg=colour, activeforeground=colour,
	                           command=lambda label=item.label: _choose_class_colour(label))
	        button.pack(side='right')
	        class_colour_buttons[item.label] = button
	    class_colours_frame.pack(fill='x', padx=8, pady=4, before=resultsShow)

	def _show_analysis_result(result):
	    nonlocal spectra_labels_exist, current_plot_data, current_plot_grid
	    global results_df, current_results_df, outputFileName
	    global mean_spectra_list, mean_spectra_dict, wavelengths, load
	    result_path = result.snapshot.output_directory / result.filename
	    with result_path.open(newline='') as file:
	        reader = csv.reader(file)
	        columns = next(reader)
	        table_rows = list(reader)

	    plot_data = spectral_analysis.prepare_plot_data(
	        result.spectra_groups, result.labels, result.labels_exist,
	    )
	    grid = result.wavelengths
	    _validate_visualisation(grid, plot_data)
	    new_image, initial_colours = _render_spectra_image(grid, plot_data)

	    # Commit the new preview only after its source table and plot are ready.
	    _replace_visualisation()
	    results_df = result.frame
	    current_results_df = result.frame.tail(1)
	    outputFileName = result.filename
	    wavelengths = grid
	    mean_spectra_list = result.mean_spectra
	    mean_spectra_dict = result.spectra_groups
	    spectra_labels_exist = result.labels_exist
	    current_plot_data = plot_data
	    current_plot_grid = grid
	    tree.delete(*tree.get_children())
	    tree['columns'] = columns
	    for column in columns:
	        tree.heading(column, text=column)
	        tree.column(column, width=100)
	    for row in table_rows:
	        tree.insert('', 'end', values=row)
	    if not analysis_table_scrollbars:
	        horizontal = ttk.Scrollbar(tab4, orient='horizontal', command=tree.xview)
	        vertical = ttk.Scrollbar(tab4, orient='vertical', command=tree.yview)
	        horizontal.pack(side='top', fill='x')
	        vertical.pack(side='right', fill='y')
	        tree.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
	        analysis_table_scrollbars.extend((horizontal, vertical))
	    tree.pack(padx=1, pady=1, fill='none', expand=False)
	    B2jump4.pack()
	    Bexport3.configure(state='normal')
	    BIexport3.configure(state='normal')
	    LResFol.config(text='Please pick a folder to save your results.', bg='white', fg='black')
	    _display_spectra_image(new_image)
	    _show_class_colour_controls(plot_data, initial_colours)
	    if not analysis_preview_scrollbars:
	        vertical = tk.Scrollbar(resultsShow, orient='vertical', command=resultsShow.yview)
	        horizontal = tk.Scrollbar(resultsShow, orient='horizontal', command=resultsShow.xview)
	        resultsShow.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
	        vertical.pack(side='right', fill='y')
	        horizontal.pack(side='bottom', fill='x')
	        analysis_preview_scrollbars.extend((vertical, horizontal))
	    _set_visualisation_ready(True)

	def _analysis_event(event, value):
	    if event == 'total':
	        progressbar2.configure(mode='determinate', maximum=value, value=0)
	        analysis_progress_label.config(text=f'0/{value} acquisitions processed')
	    elif event == 'progress':
	        progressbar2['value'] = value
	        analysis_progress_label.config(text=f'{value}/{int(progressbar2["maximum"])} acquisitions processed')
	    elif event == 'exporting':
	        analysis_progress_label.config(text=f'{value}/{value} acquisitions processed | Exporting results')

	def _analysis_finished(outcome, payload):
	    if analysis_controller.closing:
	        return
	    _set_analysis_busy(False)
	    if outcome == 'cancelled':
	        l.config(text='Analyser stopped.', bg='red')
	        analysis_progress_label.config(text='Analysis cancelled.')
	        return
	    if outcome == 'error':
	        l.config(text='Analyser stopped: please correct the reported error.', bg='red')
	        analysis_progress_label.config(text='Analysis stopped.')
	        if isinstance(payload, hyperspectral_dataset.DatasetMatchingError):
	            title = 'Hyperspectral dataset matching failed'
	        elif isinstance(payload, label_manifest.LabelManifestError):
	            title = 'Label manifest validation failed'
	        elif isinstance(payload, (ValueError, CustomIndexError)):
	            title = 'Analyser input error'
	        else:
	            title = 'Analysis error'
	        messagebox.showerror(title, str(payload), parent=application_window)
	        return
	    try:
	        _show_analysis_result(payload)
	    except Exception as error:
	        l.config(text='Analysis finished, but the results preview failed.', bg='red')
	        messagebox.showerror('Results preview failed', str(error), parent=application_window)
	        return
	    total = len(payload.frame)
	    progressbar2['value'] = total
	    analysis_progress_label.config(text=f'{total}/{total} acquisitions processed')
	    completion_status = 'Analyser finished.'
	    if payload.excluded_unlabelled_count:
	        completion_status += (
	            f' {payload.excluded_unlabelled_count} unlabelled acquisition(s) were included '
	            'in per-sample analysis and excluded from class-specific analysis.'
	        )
	    l.config(text=completion_status, bg='lightgreen')
	    messagebox.showinfo('Analysis complete', 'Analysis finished.', parent=application_window)

	def CallAnalyser():
	    if analysis_controller.active is not None or analysis_controller.closing:
	        return
	    profile = int(sensorselect.get())
	    if profile not in (0, 1, 3):
	        messagebox.showerror('Analyser input error',
	                             'Select PlantScreen VNIR, PlantScreen SWIR, or Other / broad-range spectral grid.',
	                             parent=application_window)
	        return
	    missing = [name for name, path in (
	        ('Dark HDR', globals().get('answer3')),
	        ('Data HDR', globals().get('answer3b')),
	        ('White HDR', globals().get('answer3c')),
	        ('BIL/RAW folder', globals().get('answer4')),
	        ('masks folder', globals().get('answer5')),
	    ) if not path]
	    if useorno.get() and not answerLabels:
	        missing.append('Labels file')
	    if missing:
	        messagebox.showerror('Analyser input error',
	                             'Select ' + ', '.join(missing) + ' before starting analysis.',
	                             parent=application_window)
	        return
	    output_directory = Path.cwd()
	    if getattr(sys, 'frozen', False):
	        selected_output = filedialog.askdirectory(
	            parent=application_window,
	            initialdir=str(Path.home()),
	            title='Please select the folder for automatic analysis outputs:',
	        )
	        if not selected_output:
	            return
	        output_directory = Path(selected_output)
	    snapshot = AnalysisSnapshot(
	        dark_header=answer3 or '',
	        data_header=answer3b or '',
	        white_header=answer3c or '',
	        image_directory=answer4 or '',
	        mask_directory=answer5 or '',
	        label_manifest=answerLabels or '',
	        labels_enabled=bool(useorno.get() and answerLabels),
	        sensor_type=profile,
	        metrics=tuple(metric_editor.metrics),
	        grid_wavelengths=tuple(metric_editor.queue.wavelengths),
	        output_directory=output_directory,
	        started_at=datetime.datetime.now(),
	    )
	    started = analysis_controller.start(snapshot, _analysis_event, _analysis_finished)
	    if not started:
	        return
	    _set_analysis_busy(True)
	    progressbar2.pack()
	    progressbar2.configure(mode='determinate', value=0)
	    analysis_progress_label.config(text='Preparing analysis...')
	    l.config(text='Analyser running...', bg='red')
	    l2.config(bg='white')

	application_window._bsa_analysis_controller = analysis_controller
	application_window._bsa_start_analysis = CallAnalyser
	application_window._bsa_cancel_analysis = analysis_controller.cancel
	application_window._bsa_show_analysis_result = _show_analysis_result
	#Bfolder=tk.Button(tab1,text="Select Folder with PNG Files for Segmentation",command= CallSegmentation )
	#Bstart=tk.Button(tab1,text="Start Segmentor",command= CallSegmentation )
	#Bfolder=tk.Button(left,text="Select Folder with PNG Files for Segmentation",command= CallSegmentation )
	#Bstart=tk.Button(left,text="Start Segmentor",command= CallSegmentation )
	def Model1():
		global folderingo
		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
			
		folderingo = str(resource_path('Models', 'DICOT'))
		l2.config(text='Segmentor\n\nModel Selected.')
		
		

		return folderingo
		
	def Model2():
		global folderingo
		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="lightgreen",fg='white')
		
		folderingo = str(resource_path('Models', 'GYMNOSPERM'))
		l2.config(text='Segmentor\n\nModel Selected.')
		
		return folderingo

	def Model3():	
		global folderingo

		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
			
		folderingo = str(resource_path('Models', 'MONOCOT'))
		l2.config(text='Segmentor\n\nModel Selected.')
		
		return folderingo

	def Model4():	
		my_filetypes = [('h5 files','.h5'),('all files', '.*'), ('text files', '.txt')]
		global folderingo
		importButtonCheck = True

		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
		
		folderingo = filedialog.askopenfile(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select your model:",
                                    filetypes=my_filetypes)
			
		l2.config(text='Segmentor\n\nModel Selected.')
		
		return folderingo
	'''
	if len(filelist)==0:
		#optmenu['values'] += ('IMPORT MY MODEL',)
		BModel4=tk.Button(left,text='IMPORT MY MODEL',command= Model4 )
		BModel4.pack()	

	elif len(filelist)>0:
		
		for file in filelist:
			if file == 'MONOCOT':
				BModel3=tk.Button(left,text='MONOCOT',command= Model3 )
				BModel3.pack()
			elif file == 'GYMNOSPERM':
				BModel2=tk.Button(left,text='GYMNOSPERM',command= Model2 )
				BModel2.pack()
			elif file == 'DICOT':
				BModel1=tk.Button(left,text='DICOT',command= Model1 )
				BModel1.pack()
			
		#BModel1=tk.Button(left,text=filelist[0],command= Model1 )
		#BModel2=tk.Button(left,text=filelist[1],command= Model2 )
		#BModel3=tk.Button(left,text=filelist[2],command= Model3 )
		BModel4=tk.Button(left,text='IMPORT MY MODEL',command= Model4 )
		#BModel1.pack()
		#BModel2.pack()
		#BModel3.pack()
		BModel4.pack()	
	'''
	def SingleView():
		global forSingleFile
		forSingleFileFolder = os.path.join(forSingleFile, 'originals')
		
		my_filetypes = [('PNG files', '.png'),('all files', '.*'), ('text files', '.txt')]
		answerSingle = filedialog.askopenfilename(parent=application_window,
                                    initialdir=forSingleFileFolder,
                                    title="Please select an RGB Slice\n(from originals folder within any Masks folder):",
                                    filetypes=my_filetypes)
		try:
			answerSingle = single_dialog_path(answerSingle)
		except ValueError as exc:
			messagebox.showerror("RGB Slice preview", str(exc), parent=application_window)
			return
		if answerSingle is None:
			return
		if not os.path.isfile(answerSingle):
			messagebox.showerror(
				"RGB Slice preview", "Please select an existing RGB Slice file.",
				parent=application_window,
			)
			return
									

		for widgets in Maskframe.winfo_children():
			widgets.destroy()
		
		for widgets in RGBframe.winfo_children():
			widgets.destroy()
			
		spaceforImagesMasks = tk.Label(Maskframe,text='')
		spaceforImagesMasks.pack()
		spaceforImagesMasks2 = tk.Label(Maskframe,text='')
		spaceforImagesMasks2.pack()
		spaceforImagesMasks3 = tk.Label(Maskframe,text='')
		spaceforImagesMasks3.pack()
		
		spaceforImages = tk.Label(RGBframe,text='')
		spaceforImages.pack()
		spaceforImages2 = tk.Label(RGBframe,text='')
		spaceforImages2.pack()
		spaceforImages3 = tk.Label(RGBframe,text='')
		spaceforImages3.pack()
		
		deneme = str(answerSingle)
		#print('os: '+os.getcwd())
		mainfolder = str(os.getcwd()).replace('\\','/')
		#print('forSingle: '+forSingleFileFolder)
		filenamegibi=str(forSingleFileFolder[1:]).replace('\\','/')
		#print('filenamegibi: '+filenamegibi)
		#print('deneme: '+deneme)
		#print('answer: '+answerSingle)
		deneme = str(deneme).replace(str(mainfolder),'').replace(str(filenamegibi+'/'),'')
		#print('aptal'+deneme)
		#print('aptal'+sik)
		photo_name = tk.Label(RGBframe,text='IMAGE #1: '+deneme)
		#path = os.path.join(output_dir, file_)
		try:
			image_ = decode_raster_image(answerSingle)
		except ImageInputError as exc:
			messagebox.showerror("RGB Slice preview", str(exc), parent=application_window)
			return
		n_image = image_.resize((500, 500))
		photo = PIL.ImageTk.PhotoImage(n_image)
		photo_name.pack()
		img_label = tk.Label(RGBframe, image=photo)
		img_label.photo = photo                             # <--
		img_label.pack()

		deneme = answerSingle.replace('originals/','')
		answerSingleMask = deneme[:-4]+'-PlantMask.png'
		
		#deneme  = answerSingle[:-4]+'-PlantMask.png'.replace('originals/','')
		buseferoldu=answerSingleMask.replace(str(mainfolder),'').replace(str(filenamegibi[:-9]).replace('originals/',''),'')
		photo_name = tk.Label(Maskframe,text='IMAGE #1: '+buseferoldu)
		#path = os.path.join(output_dir, file_)
		try:
			image_ = decode_raster_image(answerSingleMask, mask=True)
		except ImageInputError as exc:
			messagebox.showerror("Mask preview", str(exc), parent=application_window)
			return
		n_image = image_.resize((500, 500))
		photo = PIL.ImageTk.PhotoImage(n_image)
		photo_name.pack()
		img_label = tk.Label(Maskframe, image=photo)
		img_label.photo = photo                             # <--
		img_label.pack()
	
	Bfolder=tk.Button(left,text="Load RGB Slices",command= CallSegmentation )
	
	Bsingle=tk.Button(left,text="Preview Individual Image and Mask",command= SingleView )
	
	now = datetime.datetime.now()
	answer2 ='./Masks-'+now.strftime("%Y%m%d_%H%M%S")
	
	Bstart=tk.Button(left,text="Run Segmentation",command= CallSegmentationInit )

	#Bfolder.grid(row=3,column=0)
	#Bstart.grid(row=4,column=0)
	Bsingle.pack(side="bottom")
	Bstart.pack(side="bottom")
	Bstart["state"] = "disabled"
	Bsingle["state"] = "disabled"
	#Bstart.configure(bg="red",fg='white')
	Bfolder.pack(side="bottom")
	
	
	
	#B=tk.Button(application_window,text="Start Segmentor",command= CallSegmentation )
	#B.pack()
	
	var = tk.StringVar()
	var = 'True'
	bottomframe = ttk.Frame(tab3)
	bottomframe.pack(side='bottom', fill='x')
	upload_scroller = ttk.Frame(tab3)
	upload_scroller.pack(fill='both', expand=True)
	upload_canvas = tk.Canvas(upload_scroller, highlightthickness=0, borderwidth=0,
	                          bg=ttk.Style().lookup('TFrame', 'background'))
	upload_scrollbar = ttk.Scrollbar(upload_scroller, orient='vertical', command=upload_canvas.yview)
	upload_canvas.configure(yscrollcommand=upload_scrollbar.set)
	upload_scrollbar.pack(side='right', fill='y')
	upload_canvas.pack(side='left', fill='both', expand=True)
	upload_body = ttk.Frame(upload_canvas)
	upload_window = upload_canvas.create_window((0, 0), window=upload_body, anchor='nw')
	upload_body.bind('<Configure>', lambda _event: upload_canvas.configure(
		scrollregion=upload_canvas.bbox('all')))
	upload_canvas.bind('<Configure>', lambda event: upload_canvas.itemconfigure(
		upload_window, width=event.width))
	
	header_section = ttk.LabelFrame(upload_body, text='Header files', padding=(14, 10))
	header_section.pack(fill='x', padx=16, pady=(14, 6))
	for column in range(3):
		header_section.columnconfigure(column, weight=1)
	B2dark=tk.Button(header_section,text="DARK",command= Dark )
	B2data=tk.Button(header_section,text="DATA",command= Data )
	B2white=tk.Button(header_section,text="WHITE",command= White )
	for column, button in enumerate((B2dark, B2data, B2white)):
		button.grid(row=0, column=column, padx=5, pady=3, sticky='ew')

	profile_section = ttk.LabelFrame(upload_body, text='Analysis profile', padding=(14, 9))
	profile_section.pack(fill='x', padx=16, pady=6)
	for row, (caption, value) in enumerate((
		('PlantScreen VNIR', 0), ('PlantScreen SWIR', 1),
		('Other / broad-range spectral grid', 3),
	)):
		ttk.Radiobutton(profile_section, text=caption, variable=sensorselect, value=value).grid(
			row=row, column=0, sticky='w', pady=2)
	ttk.Label(profile_section,
	          text='Use Custom Metrics for indices appropriate to this spectral grid. '
	               'Other-camera BIL files use matching -OTHER- identities.',
	          wraplength=760, font='TkSmallCaptionFont').grid(
		row=3, column=0, sticky='w', pady=(7, 1))
	
	
	files_section = ttk.LabelFrame(upload_body, text='Hyperspectral files', padding=(14, 10))
	files_section.pack(fill='x', padx=16, pady=6)
	files_section.columnconfigure(0, weight=1)
	files_section.columnconfigure(1, weight=1)
	B2bil=tk.Button(files_section,text="Load BIL/RAW Files",command= Bil )
	B2mask=tk.Button(files_section,text="Load Segmentation Masks",command= Mask )
	B2mask_default_bg = B2mask.cget('bg')
	
	B2bil.grid(row=0, column=0, sticky='ew', padx=5, pady=3)
	B2mask.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
	labels_section = ttk.LabelFrame(upload_body, text='Labels (optional)', padding=(14, 9))
	lheader3 = ttk.Label(files_section, text='', font='TkSmallCaptionFont',
	                     foreground='#7a4d00', wraplength=760, justify='left')
	label_status = ttk.Label(labels_section, text='', font='TkSmallCaptionFont',
	                         foreground='#7a4d00', wraplength=760, justify='left')
	
	
	B2jump=tk.Button(bottomframe,text="Switch to Analyser",command= JumpToA )
	
	B2jump2=tk.Button(tab5,text="Switch Back to Analyser",command= JumpToA )
	B2jump2.pack()
	B2jump3=tk.Button(tab5,text="Switch Back to Upload Files",command= JumpToU )
	B2jump3.pack()
	
	BIexport3=tk.Button(tab5,text="Browse...",command= BrowseForExportImg )	
	LResFol2 = tk.Label(tab5, text="Save will ask for a folder, or choose one with Browse.", font="TkSmallCaptionFont")
	BIexport1=tk.Button(tab5,text="Export Mean Spectra files",command= ExportCSVMean )
	BIexport2=tk.Button(tab5,text="Export Mean Spectra files as XLSX",command= ExportXLSXMean )
	BIexportImg=tk.Button(tab5,text="Save Image",command= ExportIMG )
	BIexportImg2=tk.Button(tab5,text="Save Image as JPG",command= ExportIMGJPG )
	class_colours_frame = ttk.LabelFrame(tab5, text='Class colours', padding=4)
	
	resultsShow = Text(tab5, state='disable')
	
	resultsShow.pack(fill='both', expand=True)
	Resultsframe = tk.Frame(resultsShow)
	resultsShow.window_create("end", window=Resultsframe)
	
	
	B2jump4=tk.Button(tab4,text="Check the Spectra Visualiser",command= JumpToV )
	
	
	
	B2=tk.Button(tab4,text="Start Analyser",command= CallAnalyser )
	progressbar2 = ttk.Progressbar(tab4)
	
	tree = ttk.Treeview(tab4, show="headings")
	
	
	Bexport3=tk.Button(tab4,text="Browse...",command= BrowseForExport )	
	LResFol = tk.Label(tab4, text="", font="TkSmallCaptionFont")
	Bexport1=tk.Button(tab4,text="Export as CSV File",command= ExportCSV )
	Bexport2=tk.Button(tab4,text="Export as XLSX File",command= ExportXLSX )
	
	
	
	
	
	
	#B2=tk.Button(application_window,text="Start Analyser",command= CallAnalyser )
	
	 
	def print_selection():
		global useorno
		global answerLabels
		if useorno.get():
			_set_labels_message('Select a Labels file.')
			
			my_filetypes = [('CSV files', '.csv'),('all files', '.*'), ('text files', '.txt')]
			selected_labels = filedialog.askopenfilename(parent=application_window,
									initialdir=os.getcwd(),
									title="Please select a version 2 BSA label manifest:",
									filetypes=my_filetypes)
			if selected_labels == '':
				if answerLabels:
					_set_labels_message('Labels file: ' + Path(answerLabels).name)
				else:
					useorno.set(False)
					_set_labels_message('No Labels file selected.')
			else:
				try:
					inspect_selected_label_manifest(selected_labels)
				except label_manifest.LabelManifestError as exc:
					useorno.set(bool(answerLabels))
					_set_labels_message(
						'The selected CSV is invalid; the previous Labels file remains selected.'
						if answerLabels else 'The selected CSV is not a valid version 2 BSA label manifest.'
					)
					messagebox.showerror(
						'Label manifest not accepted',
						str(exc),
						parent=application_window,
					)
				else:
					answerLabels = selected_labels
					_set_labels_message('Labels file: ' + Path(answerLabels).name)
		else:
			answerLabels = ''
			_set_labels_message('')
		#account.config(text='This is possible.' + str(useorno.get()) + ' Files')
		#B2.pack()
		
	#lconv = tk.Label(application_window, bg='white', width=200, text='Convert HDR to ENVI',padx=10, pady=10)
	#lconv.pack()
	#B3=tk.Button(application_window,text="Start HDRtoENVI Convertor",command= CallConventor )
	#B3.pack();

	hdr_panel = HDRCreatorPanel(tab2, application_window, open_in_file_manager)
	hdr_panel.pack(fill='both', expand=True)
	application_window._bsa_hdr_creator = hdr_panel
	l = tk.Label(tab4, bg='white', width=200, text='Analyser',padx=10, pady=10, font="TkHeadingFont")
	#l = tk.Label(application_window, bg='white', width=200, text='Analyser',padx=10, pady=10)

	l.pack()
	metric_editor = CustomMetricEditor(upload_body, application_window)
	application_window._bsa_custom_metric_editor = metric_editor
	analysis_progress_label = ttk.Label(tab4, text='Analysis ready', font='TkSmallCaptionFont')
	analysis_progress_label.pack()

	B2.pack()
	LResFol.pack()
	Bexport3.pack()
	Bexport1.pack()
	Bexport2.pack()
	

	LResFol2.pack()
	BIexport3.pack()
	BIexport1.pack()
	#BIexport2.pack()
	BIexportImg.pack()
	#BIexportImg2.pack()
	
	BIexport1["state"] = "disabled"
	#BIexport2["state"] = "disabled"
	BIexportImg["state"] = "disabled"
	#BIexportImg2["state"] = "disabled"
	
	B2["state"] = "disabled"
	Bexport1["state"] = "disabled"
	Bexport2["state"] = "disabled"
	Bexport3["state"] = "disabled"
	B2jump.forget()
	
	


	
	
	#r1 = tk.Radiobutton(tab3, text='Start Analyser without Segmentor (if you have Masks ready)', variable=var, value='True', command=print_selection)
	#r1.pack()
	#r2 = tk.Radiobutton(tab3, text='Use HDR Files', variable=var, value='HDR', command=print_selection)
	#r2.pack()
	
	labels_section.pack(fill='x', padx=16, pady=(6, 12))
	checkLabels = ttk.Label(labels_section,text='Use a version 2 Labels file for class-specific results:')
	checkLabels.pack(anchor='w', padx=8, pady=(3, 5))
	

	#useorno = tk.IntVar(value=0)
	#vars_list.append(var)
	#tk.Radiobutton(tab3, text="Yes", variable=var, value=1).pack(anchor='center',side="left")
	#tk.Radiobutton(tab3, text="No", variable=var, value=0).pack(anchor='center',side="left")
	label_choices = ttk.Frame(labels_section)
	label_choices.pack(anchor='w', padx=8, pady=(0, 3))
	ttk.Radiobutton(label_choices, text="Yes", variable=useorno, value=True,command=print_selection).pack(side='left')
	ttk.Radiobutton(label_choices, text="No", variable=useorno, value=False,command=print_selection).pack(side='left', padx=(14, 0))


	def restart_program():
		"""Restarts the current program.
		Note: this function does not return. Any cleanup action (like
		saving data) must be done before calling this function."""
		python = sys.executable
		os.execl(python, python, * sys.argv)
	
	def close_program():
		nonlocal segmentation_closing
		if segmentation_closing:
			return
		segmentation_closing = True
		analysis_controller.close()
		active_thread = None
		if segmentation_job is not None:
			segmentation_job['cancelled'].set()
			active_thread = segmentation_job['thread']
			progressbar.stop()
			if segmentation_job['after_id'] is not None:
				application_window.after_cancel(segmentation_job['after_id'])
		for callback_id in application_window.tk.call('after', 'info'):
			application_window.tk.call('after', 'cancel', callback_id)
		application_window.quit()
		application_window.destroy()
		# TensorFlow must finish its current call before interpreter shutdown.
		if active_thread is not None and active_thread is not threading.current_thread():
			active_thread.join()

	LogoImage = PIL.Image.open(resource_path("BSA_logo.png"))	
	LogoImageResize = LogoImage.resize((50, 50))
	Logo = PIL.ImageTk.PhotoImage(LogoImageResize)

	#Label(application_window, width=275, text='brought to you by Jason Walsh from UCD', image=Logo, compound='left' ,padx=10, pady=10,anchor="w", justify="left").pack()	
		
	##LogoImage = PIL.Image.open("BSA_logo.png")
	##LogoImageResize = LogoImage.resize((100, 100))
	##Logo = PIL.ImageTk.PhotoImage(LogoImageResize)
	
	#LogoImage = PhotoImage(file=)	
	#LogoImage = LogoImage.zoom(25) #with 250, I ended up running out of memory
	#LogoImage = LogoImage.subsample(32) 	
	##lcloseImg = tk.Label(application_window, justify="left", image=Logo)
	##lcloseImg.pack()
	#lclose = tk.Label(application_window, text='brought to you by Jason Walsh from UCD', justify="left")
	#lclose = tk.Label(application_window, bg='white', width=200, text='brought to you by Jason Walsh from UCD',padx=10, pady=10,anchor="w", justify="left")
	#lclose = tk.Label(application_window, anchor="w", justify="left", bg='lightgreen', width=300, text='brought to you by Jason Walsh from UCD', compound='left',image=Logo)
	# lclose = tk.Label(application_window, justify="left", bg='lightgreen', width=300, text='brought to you by Jason Walsh from UCD', compound='left',image=Logo)
	#lclose.place(relx = 0.0, rely = 1.0, anchor ='sw')
	#lclose = tk.Label(application_window, width=300, text='brought to you by Jason Walsh from UCD', compound='left' ,anchor="w", justify="left", image=Logo )
	# lclose.pack(fill="x", expand=False)
	
	#Button(application_window, text="Restart", command=restart_program).pack()
	#Button(application_window, text="Close", command=close_program).pack()
	
	btton1=Button(application_window,text='Restart',width=15,height=1, command=restart_program)
	btton1.pack(side='left', anchor='e', expand=True)
	btton2=Button(application_window,text='Exit',width=15,height=1, command=close_program)
	btton2.pack(side='right', anchor='w', expand=True)

	#def showlabel():
	#	account.place(relx=0.01, rely=0.125)  # move on screen
		
	#account_frame = tk.Frame(application_window, width=300, height=700, bg='#9dd7f5')
	#account_frame = tk.Frame(application_window, bg='#9dd7f5')
	#account_frame.pack(side='left')
	#add_account_button = tk.Button(account_frame, text="Add Account", font='Courier 15', command=showlabel)
	#add_account_button.place(relx=0.26, rely=0.03)
	#account = tk.Label(account_frame, text="Account Nr1", font='Helvetica 20')
	#account.place(relx=10.01, rely=0.125)  # off screen
	

	application_window.config(menu=menubar)
	application_window.protocol('WM_DELETE_WINDOW', close_program)
	application_window._bsa_close_program = close_program
	show_about_page()	
	application_window.mainloop()
	
	

	#application_window.mainloop()	
	
def run():
	"""Launch the authoritative recovered BSA GUI directly."""
	main()


if __name__ == "__main__":
	run()
	




	







	
