import sys
import os
import cv2
import csv
import shutil
import time

from io import BytesIO


import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from custom_metrics import iou, dice_coef, dice_loss
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText


import os
from tkinter import *
from tkinter import ttk  

import datetime

import PIL.Image
import PIL.ImageTk

import os
import re
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

import webbrowser

from cryptography.fernet import Fernet
#import webview
from misc import export_results, export_spectra_data, calculate_custom_index, categorize_file

application_window = tk.Tk()
#application_window.geometry('1024x600')
#application_window.attributes('-fullscreen',True)

application_window.geometry('800x600')
#application_window.geometry('600x400')
#application_window.geometry('1200x800')

#getting screen width and height of display
#width= application_window.winfo_screenwidth() 
#height= application_window.winfo_screenheight()
#setting tkinter window size
#application_window.geometry("%dx%d" % (width, height))

application_window.title('Botanic Spectrum Analyser v5.1')


application_window.iconbitmap("BSA_logo.ico")

custom_index_list = []  # List of dicts like: {"name": str, "bands": {"Var1": int, "Var2": int}, "formula": str}
answer = None
folderingo = None
RGBCheck = False
ModelCheck = False
importButtonCheck = False
yahu = tk.StringVar()
forSingleFile = None
answerLabels = ''

DarkCheck = False
DataCheck = False
BilCheck = False
WhiteCheck = False
MaskCheck = False
useorno = tk.BooleanVar(value=False)
#sensorselect = 0
sensorselect = tk.IntVar(value=0)

#useorno False

import os
import sys
import json
import tkinter as tk
from tkinter import messagebox, ttk
from PyPDF2 import PdfReader

SECRET_KEY = b'Ft240KneH8tvqLr7bsv2I3WFCYHZdFMUocTkRlFVlpg='
cipher = Fernet(SECRET_KEY)

# Constants for user agreement storage
AGREEMENT_DIR = os.path.join(os.path.expanduser("~"), ".bsa")  # Hidden folder
AGREEMENT_FILE = os.path.join(AGREEMENT_DIR, "user_agreement.json")
PDF_FILE = "terms_and_conditions.pdf"

os.makedirs(AGREEMENT_DIR, exist_ok=True)

def encrypt_data(data):
    """Encrypts the data before saving."""
    return cipher.encrypt(data.encode())

def decrypt_data(encrypted_data):
    """Decrypts the stored data."""
    return cipher.decrypt(encrypted_data).decode()

def restart_program():
    """Restarts the current program."""
    python = sys.executable
    os.execl(python, python, *sys.argv)

def has_user_agreed():
    """Check if the user has previously agreed to the terms."""
    if os.path.exists(AGREEMENT_FILE):
        try:
            with open(AGREEMENT_FILE, "rb") as f:  # Read as binary
                encrypted_data = f.read()
                decrypted_data = decrypt_data(encrypted_data)
                data = json.loads(decrypted_data)
                return data.get("agreed", False)
        except Exception as e:
            print(f"Error decrypting user agreement: {e}")
            return False  # If error, force the user to re-agree
    return False

def save_user_agreement(name, email):
    """Encrypt and save user agreement details."""
    data = json.dumps({"name": name, "email": email, "agreed": True})
    encrypted_data = encrypt_data(data)

    with open(AGREEMENT_FILE, "wb") as f:  # Save as binary
        f.write(encrypted_data)

def open_pdf():
    """Display the PDF file."""
    pdf_window = tk.Toplevel()
    pdf_window.title("Terms and Conditions")
    pdf_window.geometry("600x400")
    pdf_window.grab_set()  # Ensure this window is interactive

    text_area = tk.Text(pdf_window, wrap=tk.WORD, width=80, height=20)
    text_area.pack(expand=True, fill=tk.BOTH)

    try:
        with open(PDF_FILE, "rb") as f:
            pdf_reader = PdfReader(f)
            pdf_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            text_area.insert(tk.END, pdf_text)
    except Exception as e:
        text_area.insert(tk.END, f"Error loading PDF: {e}")
    
    text_area.config(state=tk.DISABLED)

    # Add Close Button
    close_button = tk.Button(pdf_window, text="Close", command=pdf_window.destroy)
    close_button.pack(pady=10)

    pdf_window.transient(application_window)  # Tie the window to the main app
    pdf_window.focus_force()  # Bring focus to this window

def show_splash():
    """Display the splash screen for user agreement."""
    def submit():
        name = name_entry.get().strip()
        email = email_entry.get().strip()
        agreed = agree_var.get()

        if not name or not email:
            messagebox.showerror("Error", "Name and Email are required!")
            return
        if not agreed:
            messagebox.showerror("Error", "You must agree to continue!")
            return

        save_user_agreement(name, email)

        splash.destroy()
        application_window.quit()  # Ensure the old GUI fully quits
        application_window.destroy()

        restart_program()  # Restart after agreement

    splash = tk.Toplevel()
    splash.title("User Agreement")
    splash.geometry("400x300")
    splash.grab_set()  # Prevent interaction with other windows

    # ✅ Set window icon
    try:
        splash.iconbitmap("BSA_logo.ico")  # Works on Windows
    except:
        logo_img = PIL.ImageTk.PhotoImage(PIL.Image.open("BSA_logo.png"))  # Use PNG as a backup
        splash.tk.call("wm", "iconphoto", splash._w, logo_img)  # Works on Linux/MacOS

    ttk.Label(splash, text="Please read and agree to the terms:", font=("Arial", 12)).pack(pady=10)
    ttk.Button(splash, text="Read PDF", command=open_pdf).pack(pady=5)

    ttk.Label(splash, text="Full Name:").pack()
    name_entry = ttk.Entry(splash, width=40)
    name_entry.pack()

    ttk.Label(splash, text="Email:").pack()
    email_entry = ttk.Entry(splash, width=40)
    email_entry.pack()

    agree_var = tk.BooleanVar()
    agree_check = ttk.Checkbutton(splash, text="I agree to the terms", variable=agree_var)
    agree_check.pack(pady=5)

    submit_button = ttk.Button(splash, text="Submit", command=submit)
    submit_button.pack(pady=10)

    splash.mainloop()

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

def show_about_page():
    """Create a splash screen with structured content."""
    about_window = tk.Toplevel()
    about_window.title("About")
    about_window.geometry("400x500")
    about_window.resizable(False, False)
	
    try:
        about_window.iconbitmap("BSA_logo.ico")  # Works for Windows
    except:
        logo_img = PIL.ImageTk.PhotoImage(PIL.Image.open("BSA_logo.png"))  # Use PNG as a backup
        about_window.tk.call("wm", "iconphoto", about_window._w, logo_img)  # Works on Linux/MacOS

    # Load the logo
    try:
        logo_image = PIL.Image.open("BSA_logo.png")
        logo_image = logo_image.resize((100, 100), PIL.Image.LANCZOS)
        logo_photo = PIL.ImageTk.PhotoImage(logo_image)
    except Exception as e:
        print(f"Error loading logo: {e}")
        logo_photo = None

    # Load the logo
    try:
        logo_image = PIL.Image.open("BSA_logo.png")
        logo_image = logo_image.resize((100, 100), PIL.Image.LANCZOS)
        logo_photo = PIL.ImageTk.PhotoImage(logo_image)
    except Exception as e:
        print(f"Error loading logo: {e}")
        logo_photo = None

    # Logo Label
    if logo_photo:
        logo_label = tk.Label(about_window, image=logo_photo)
        logo_label.image = logo_photo  # Keep a reference
        logo_label.pack(pady=(20, 5))  # Adjust padding for spacing

    # Title Text
    label_title = ttk.Label(about_window, text="Botanical Spectrum Analyser", font=("Arial", 12, "bold"))
    label_title.pack(pady=5)

    # Scrollable Text Widget (Properly formatted paragraph)
    about_text = """Welcome to Botanical Spectrum Analyser - BSA, a user-friendly tool for biologists, breeders, and plant scientists to easily analyse and interpret plant hyperspectral images.

The BSA is a free and open-source software developed for segmenting and analysing hyperspectral imaging. Hyperspectral imaging collects the spectrum for each pixel within a plant image, enabling the detection of stresses such as diseases, abiotic responses, crop monitoring, nutrient levels, etc.

Due to the complexity of hypercubes (3D datasets comprising two spatial dimensions i.e., plant image, and one wavelength dimension), hyperspectral image analysis and segmentation are challenging for non-technical users. 

BSA software is built in Python using Tkinter for the graphical user interface, OpenCV for image processing, and TensorFlow for deep learning operations.

The BSA software enables the accurate segmentation of hyperspectral images and provides real-time analysis of hypercubes. These include an array of pre-calculated or user-specified vegetation indices (e.g., NDVI, NDWI, etc.) and a visualisation of the mean spectra graph of each hypercube."""

    # ScrolledText widget to hold the paragraph text properly
    text_widget = ScrolledText(about_window, wrap=tk.WORD, width=45, height=10, font=("Arial", 9))
    text_widget.insert(tk.END, about_text)  # Insert text into the widget
    text_widget.config(state=tk.DISABLED)  # Make it read-only
    text_widget.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    # Version Text
    label_version = ttk.Label(about_window, text="Version: 1.0", font=("Arial", 9))
    label_version.pack(pady=5)

    # Close Button
    close_button = ttk.Button(about_window, text="Close", command=about_window.destroy)
    close_button.pack(pady=10)

menubar = Menu(application_window)
# About menu
aboutmenu = Menu(menubar, tearoff=0)
aboutmenu.add_command(label="About BSA", command=show_about_page)  # Calls show_about_page()
menubar.add_cascade(label="About", menu=aboutmenu)

# Help menu
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Show Video Tutorial", command=lambda: None)  # Placeholder function
helpmenu.add_command(label="Show Manual", command=lambda: None)  # Placeholder function
menubar.add_cascade(label="Help", menu=helpmenu)

# Set the updated menubar
application_window.config(menu=menubar)

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
		if not os.path.exists(path):
			os.makedirs(path)

	def save_results(self, image, y_pred, save_image_path):
		"""
		Save the segmentation results.
		
		Args:
			image (np.ndarray): Input image.
			y_pred (np.ndarray): Predicted segmentation mask.
			save_image_path (str): Path to save the image.
		"""
		try:
			file_name = os.path.basename(save_image_path)
			save_dir = os.path.dirname(save_image_path)

			# Append "-PlantMask" to the file name
			if not file_name.endswith("-PlantMask.png"):
				file_name = file_name.split('.')[0] + "-PlantMask.png"

			save_path = os.path.join(save_dir, file_name)

			# Convert the mask to uint8 if necessary
			if y_pred.dtype != np.uint8:
				y_pred = (y_pred * 255).astype(np.uint8)

			cv2.imwrite(save_path, y_pred)

		except Exception as e:
			print(f"Error occurred while saving the image: {str(e)}")

			
	def pad_image(self, image, new_size=(512, 512)):
		"""Pad an image up to the new size."""
		old_size = image.shape[:2]
		delta_w = new_size[1] - old_size[1]
		delta_h = new_size[0] - old_size[0]

		top, bottom = delta_h // 2, delta_h - (delta_h // 2)
		left, right = delta_w // 2, delta_w - (delta_w // 2)

		padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
		return padded_image

	def unpad_image(self, image, old_size):
		"""Unpad an image to the old size."""
		height, width = old_size

		top = (image.shape[0] - height) // 2
		bottom = top + height
		left = (image.shape[1] - width) // 2
		right = left + width

		return image[top:bottom, left:right]

	def resize_image(self, image, new_size=(512, 512)):
		"""Resize an image while maintaining the aspect ratio."""
		old_size = image.shape[:2]
		ratio = min(new_size[0]/old_size[0], new_size[1]/old_size[1])
		intermediate_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
		resized_image = cv2.resize(image, intermediate_size, interpolation=cv2.INTER_AREA)

		delta_w = new_size[1] - intermediate_size[0]
		delta_h = new_size[0] - intermediate_size[1]
		top, bottom = delta_h // 2, delta_h - (delta_h // 2)
		left, right = delta_w // 2, delta_w - (delta_w // 2)

		return cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

						
	def run_segmentation(self):
		"""
		Run the image segmentation process.
		"""
		# Seeding
		np.random.seed(42)
		tf.random.set_seed(42)

		# Loading model
		with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
			model = tf.keras.models.load_model(self.model_path)

		# Load the dataset
		test_x = sorted([os.path.join(self.input_dir, file_name) for file_name in os.listdir(self.input_dir)])

		# Print the file paths to check if they are valid
		print(test_x)

		self.progressbar["mode"] = "determinate"
		self.progressbar["value"] = 0
		self.progressbar["maximum"] = len(test_x)

		progress_text_label = ttk.Label(self.left, text="Starting segmentation...", font=("Arial", 10))
		progress_text_label.pack()

		start_time = time.time()

		# Create a tqdm progress bar
		for i, x in enumerate(test_x):
			
			self.progressbar["value"] = i + 1

			elapsed = time.time() - start_time
			avg_time = elapsed / (i + 1)
			remaining = avg_time * (len(test_x) - i - 1)

			progress_text_label.config(text=f"{i + 1}/{len(test_x)} images segmented | ETA: {int(remaining)}s")
			application_window.update_idletasks()

			self.progressbar.update()
			application_window.update()
			application_window.update_idletasks()
			
			try:
				file_name = os.path.basename(x)

				# Reading the image
				image = cv2.imread(x, cv2.IMREAD_COLOR)

				# Store original size
				original_size = image.shape[:2]  # (height, width)

				# Resize or pad the image as needed
				processed_image = image
				if max(original_size) > 512:
					processed_image = self.resize_image(image)  # Resize if too large
				elif min(original_size) < 512:
					processed_image = self.pad_image(image)  # Pad if too small

				# Prepare the image for the model
				x = processed_image / 255.0
				x = np.expand_dims(x, axis=0)

				# Prediction
				y_pred = model.predict(x)[0]
				y_pred = np.squeeze(y_pred, axis=-1)

				# Resize or unpad the prediction to match the original size
				if max(original_size) > 512:
					y_pred = cv2.resize(y_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
				else:
					y_pred = self.unpad_image(y_pred, original_size)

				# Save the prediction
				save_image_path = os.path.join(self.output_dir, f"{file_name.split('.')[0]}-PlantMask.png")
				self.save_results(image, y_pred, save_image_path)

			except Exception as e:
				print(f"Error occurred while processing image: {x}")
				print(f"Error message: {str(e)}")
			
		progress_text_label.config(text=f"Done: {len(test_x)}/{len(test_x)} images segmented.")
	
def CallHDRCreatorExtra(samples, lines, bands, bits, byteorder, wavelength_range, file_name,folder,label):
	
	
		label.config(text='HDR Creator started...',bg='red')

	
		# Function to generate wavelengths
		def generate_wavelengths(start, end, steps):
			wavelength_list = [start]
			step_size = (end - start) / (steps - 1)
			for i in range(1, steps):
				next_wavelength = round(start + i * step_size, 2)
				wavelength_list.append(next_wavelength)
			return wavelength_list

		wavelengths = generate_wavelengths(wavelength_range[0], wavelength_range[1], bands)
		formatted_wavelengths = ", ".join([f"{val:.2f}" for val in wavelengths])
		hdr_template = f'''ENVI
description = {{ }}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = {bits}
interleave = BIL
sensor type = Unknown
byte order = {byteorder}
wavelength units = Unknown
wavelength = {{
{formatted_wavelengths}
}}'''

	
	
		print(samples)
		print(lines)
		print(bands)
		print(bits)
		print(byteorder)
		print(wavelength_range)
		print(file_name)
		print(folder)
		
		#print ('I AM HERE')
		with open(f""+folder+"/"+file_name+".hdr", "w") as f:
			f.write(hdr_template)
		
		label.config(text='HDR Creator finished...Your file is ready at \n '+folder+'/'+file_name+'.hdr',bg='lightgreen')
		os.startfile(folder)

	
	
	
#def CallAnalyser(hdrorenvifold,bilfold,marksfol):
def CallAnalyserExtra(hdrdark,hdrdata,hdrwhite,bilfold,marksfold,labelGUI,tab,tree,tc,tabx,button,Resultsframex,resultsShowx,bex1,bex2,bex3,LResFolx,progressbar2,biex3):
	#os.system('python segmentor.py')
	#print(hdrorenvifold)
	print("DEBUG: CallAnalyserExtra() started.")  # Debugging
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

	
	print(hdrdark)
	print(hdrdata)
	print(hdrwhite)
	print(bilfold)
	print(marksfold)
	
	labelGUI.config(text='Analyser started...',bg='red')
	progressbar2.pack()
	progressbar2["mode"] = "determinate"
	progressbar2["value"] = 0

	#input_dir = "RGB Images"
	#input_dir = infold
	#output_dir = "Masks"
	#output_dir = outfold
	print ('hoppaanalyserExtraCalled')
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

	# Create the custom index list
	custom_indices = [] 

	# Example user input (replace this with GUI input)
	#user_index_name = "NDVI Custom"  # Replace with user input for the index name
	#user_formula = "(NIR - RED) / (NIR + RED)"  # User-defined formula
	#user_formula = "(RED - NIR)"  # User-defined formula
	#user_bands = {"NIR": 395, "RED": 281}  # Mapping of band names to indices (I am using NDVI as the example even though I am already including it :-) )
	
	# List all files in the directory
	imgFiles = sorted([file for file in os.listdir(img_directory_path) if file.endswith('.bil') or file.endswith('.raw')])
	maskFiles = sorted([file for file in os.listdir(mask_directory_path) if file.endswith('.png') or file.endswith('.jpg')])

	# Initialize arrays to store bil file names for Dark, Data, and White calibration
	dark_img_file_names = []
	data_img_file_names = []
	white_img_file_names = []

	# Initialize array to store file names for masks
	mask_file_names = []

	# Categorize files based on their names
	for fileName in imgFiles:
		category = categorize_file(fileName)
		if category == 'dark':
			dark_img_file_names.append(fileName)
		elif category == 'data':
			data_img_file_names.append(fileName)
		elif category == 'white':
			white_img_file_names.append(fileName)

	# Loop through each .bil file and categorize the file names
	for maskFileName in maskFiles:
		mask_file_names.append(maskFileName)

	# Count the number of files with the .bil extension
	numFiles = len(imgFiles)

	# Define the number of sets to process (X)
	numSets = numFiles // 3

	progressbar2["maximum"] = numSets  # ✅ Now numSets is defined

	progress_text_label = ttk.Label(tab, text="Starting analysis...", font=("Arial", 10))
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
	print ("Current date and time : ")
	print (now.strftime("%Y-%m-%d %H:%M:%S"))
		
	# Define the filename for the output file (CSV or XLSX)
	#outputFileName = 'Results' + ('.csv' if output_format == 'csv' else '.xlsx')
	outputFileName = 'Results-'+now.strftime("%Y%m%d_%H%M%S")+('.csv' if output_format == 'csv' else '.xlsx')
	# Change the filename and extension as needed

	# Check if Labels.csv exists
	#labels_file_path = 'Labels.csv'
	labels_file_path = answerLabels
	labels_exist = os.path.isfile(labels_file_path)

	# Initialize labels list
	labels = []

	# Read the CSV file with labels if it exists
	if useorno.get() and answerLabels != '':
	#if labels_exist:
		print ('use'+str(useorno.get()))
		labels_file_path = answerLabels
		print ('folder:'+str(labels_file_path))
		with open(labels_file_path, 'r') as csvfile:
			csv_reader = csv.reader(csvfile)
			for row in csv_reader:
				labels.append(row[0])
	else:
		print ('dontuse'+str(useorno.get()))
		print("Labels not found. Proceeding without labels.")

	# Function to extract wavelengths from the ENVI header
	def extract_wavelengths(envi_header_path):
		wavelengths = []
		with open(envi_header_path, 'r') as f:
			content = f.read()
			match = re.search(r'wavelength\s*=\s*{([^}]*)}', content, re.DOTALL)
			if match:
				wavelengths_str = match.group(1).strip()
				wavelengths = list(map(float, wavelengths_str.split(',')))
			else:
				print("No wavelengths found in the file.")
		return wavelengths

	# Extract the wavelengths being used into memory
	extracted_wavelengths = extract_wavelengths(data_hdr_path)

	# Check that the wavelengths are ok
	if extracted_wavelengths:
		wavelengths = np.array(extracted_wavelengths)  # Convert to numpy array for plotting
	else:
		print("No wavelengths were extracted. Please check the file format.")

	# Initialize an empty list to store the mean spectra
	mean_spectra_list = []

	# Initialize dictionaries to store mean spectra for each category
	mean_spectra_dict = {label: [] for label in labels}

	
	# Example: Separate stats dictionaries for VNIR and SWIR
	vnir_stats = {}
	swir_stats = {}
	combined_stats = {}

	# 0 for VNIR, 1 for SWIR, 2 for both
	sensor_type = 1  # Replace with user input or configuration
	sensor_type = sensorselect.get()
	
	print ('sensor select: '+str(sensor_type))

	##########################################################################################
	#                                                                                        #
	#                                     Program Run                                        #
	#                                                                                        #
	##########################################################################################

	print("Will use these custom formulas:", custom_index_list)
	for setIndex in tqdm(range(numSets), desc="Processing"):
	
		progressbar2["value"] = setIndex + 1

		elapsed = time.time() - start_time
		avg_time = elapsed / (setIndex + 1)
		remaining = avg_time * (numSets - setIndex - 1)

		progress_text_label.config(text=f"{setIndex + 1}/{numSets} images processed | ETA: {int(remaining)}s")
		application_window.update_idletasks()
		# Construct file paths for Dark, Data, White calibration bil files
		dark_img_path = os.path.join(img_directory_path, dark_img_file_names[setIndex])
		# print("Dark BIL file path: ", dark_bil_path)

		data_img_path = os.path.join(img_directory_path, data_img_file_names[setIndex])
		# print("Data BIL file path: ", data_bil_path)

		white_img_path = os.path.join(img_directory_path, white_img_file_names[setIndex])
		# print("White BIL file path: ", white_bil_path)

		# Construct file paths for masks
		mask_path = os.path.join(mask_directory_path, mask_file_names[setIndex])
		# print("Mask file path: ", mask_path)

		# Load in the data using envi.open from spectral python
		dark = envi.open(dark_hdr_path, dark_img_path)
		data = envi.open(data_hdr_path, data_img_path)
		white = envi.open(white_hdr_path, white_img_path)

		# Convert spectral python objects to ndarrays
		data_arr = data.load()
		# print("Data Size: ", data_arr.shape)

		dark_arr = dark.load()
		# print("Dark Size: ", dark_arr.shape)

		white_arr = white.load()
		# print("White Size: ", white_arr.shape)

		# Extract the shape of the data_arr object
		x, y, z = data_arr.shape

		# Initialize calibrated data to be the same size as the data array
		calibrated_data = np.zeros_like(data_arr)

		# Loop through each band and apply pixel-wise calibration
		for band in range(z):
			calibrated_data[:, :, band] = (data_arr[:, :, band] - dark_arr[0, :, band]) / white_arr[0, :, band]

		# IMPORTANT - Convert the calibrated_data object to a numpy array as after calibration it will an ImageArray object.
		calibrated_data_arr = np.array(calibrated_data)

		##########################################################################################
		#                                                                                        #
		#                                       Masking                                          #
		#                                                                                        #
		##########################################################################################

		# Read in the binary mask used for segmentation and convert to grayscale
		imagePath = mask_path
		mask = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

		# Create a nan mask equivalent of the grayscaled image
		nanMask = np.where(mask == 255, 1.0, np.nan)

		# Initialize data masked to be the same size as the calibrated data array
		data_masked = np.zeros_like(calibrated_data_arr)

		# Loop through all bands (z) of the calibrated data and multiply each slice by the nanmask
		for i in range(z):
			data_masked[:, :, i] = calibrated_data_arr[:, :, i] * nanMask
		
		##########################################################################################
		#                                                                                        #
		#                                     Mean Spectrum                                      #
		#                                                                                        #
		##########################################################################################

		# Use label if available, otherwise skip label-specific processing
		label = labels[setIndex] if labels_exist and setIndex < len(labels) else None

		mean_spectrum = np.nanmean(data_masked, axis=(0, 1))
		mean_spectra_list.append(mean_spectrum)

		# ✅ Store each image separately in dictionary
		image_label = f"Image {setIndex+1}"
		mean_spectra_dict[image_label] = [mean_spectrum]

		# ✅ If labels exist, also store by class
		if labels_exist and setIndex < len(labels):
			label = labels[setIndex]
			if label not in mean_spectra_dict:
				mean_spectra_dict[label] = []
			mean_spectra_dict[label].append(mean_spectrum)

		print(f"DEBUG: Stored Mean Spectrum for {image_label} -> {mean_spectrum}")

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

		# Conditional calculation based on sensor type
		if sensor_type in [0, 2]:  # VNIR or both
			print (' VNIR or BOTH sensor select: '+str(sensor_type))
			
			NIR = data_masked[:, :, 395]
			RED = data_masked[:, :, 281]

			NDVI = 0
			NDVI = np.divide(NIR - RED, NIR + RED)

			meanNDVI = np.nanmean(NDVI)
			minNDVI = np.nanmin(NDVI)
			maxNDVI = np.nanmax(NDVI)
			medianNDVI = np.nanmedian(NDVI)
			stdNDVI = np.nanstd(NDVI)

			##########################################################################################

			FirstBandPRI = data_masked[:, :, 159]
			SecondBandPRI = data_masked[:, :, 193]

			PRI = 0
			PRI = np.divide(FirstBandPRI - SecondBandPRI, FirstBandPRI + SecondBandPRI)

			meanPRI = np.nanmean(PRI)
			minPRI = np.nanmin(PRI)
			maxPRI = np.nanmax(PRI)
			medianPRI = np.nanmedian(PRI)
			stdPRI = np.nanstd(PRI)

			##########################################################################################

			FirstBandPSRI = data_masked[:, :, 290]
			SecondBandPSRI = data_masked[:, :, 131]
			ThirdBandPSRI = data_masked[:, :, 351]

			PSRI = 0
			PSRI = np.divide(FirstBandPSRI - SecondBandPSRI, ThirdBandPSRI)

			meanPSRI = np.nanmean(PSRI)
			minPSRI = np.nanmin(PSRI)
			maxPSRI = np.nanmax(PSRI)
			medianPSRI = np.nanmedian(PSRI)
			stdPSRI = np.nanstd(PSRI)

			##########################################################################################

			FirstBandSIPI = data_masked[:, :, 386]
			SecondBandSIPI = data_masked[:, :, 88]
			ThirdBandSIPI = data_masked[:, :, 263]

			SIPI = 0
			SIPI = np.divide(FirstBandSIPI - SecondBandSIPI, FirstBandSIPI + ThirdBandSIPI)

			meanSIPI = np.nanmean(SIPI)
			minSIPI = np.nanmin(SIPI)
			maxSIPI = np.nanmax(SIPI)
			medianSIPI = np.nanmedian(SIPI)
			stdSIPI = np.nanstd(SIPI)

			##########################################################################################

			NIR = data_masked[:, :, 386]
			REDE = data_masked[:, :, 325]

			NDRE = 0
			NDRE = np.divide(NIR - REDE, NIR + REDE)

			meanNDRE = np.nanmean(NDRE)
			minNDRE = np.nanmin(NDRE)
			maxNDRE = np.nanmax(NDRE)
			medianNDRE = np.nanmedian(NDRE)
			stdNDRE = np.nanstd(NDRE)

			##########################################################################################

			FirstBandWP1 = data_masked[:, :, 276]
			SecondBandWP1 = data_masked[:, :, 320]

			WP1 = 0
			WP1 = np.divide(FirstBandWP1 - SecondBandWP1, SecondBandWP1)

			meanWP1 = np.nanmean(WP1)
			minWP1 = np.nanmin(WP1)
			maxWP1 = np.nanmax(WP1)
			medianWP1 = np.nanmedian(WP1)
			stdWP1 = np.nanstd(WP1)

			##########################################################################################

			# Create a dictionary with the statistics for this image set
			vnir_stats = {
				'File Name': dataFileNameWithoutExt,
				'Label': label,
				'NDVI Mean': meanNDVI,
				'NDVI Min': minNDVI,
				'NDVI Max': maxNDVI,
				'NDVI Median': medianNDVI,
				'NDVI Std': stdNDVI,
				'PRI Mean': meanPRI,
				'PRI Min': minPRI,
				'PRI Max': maxPRI,
				'PRI Median': medianPRI,
				'PRI Std': stdPRI,
				'PSRI Mean': meanPSRI,
				'PSRI Min': minPSRI,
				'PSRI Max': maxPSRI,
				'PSRI Median': medianPSRI,
				'PSRI Std': stdPSRI,
				'SIPI Mean': meanSIPI,
				'SIPI Min': minSIPI,
				'SIPI Max': maxSIPI,
				'SIPI Median': medianSIPI,
				'SIPI Std': stdSIPI,
				'NDRE Mean': meanNDRE,
				'NDRE Min': minNDRE,
				'NDRE Max': maxNDRE,
				'NDRE Median': medianNDRE,
				'NDRE Std': stdNDRE,
				'WP1 Mean': meanWP1,
				'WP1 Min': minWP1,
				'WP1 Max': maxWP1,
				'WP1 Median': medianWP1,
				'WP1 Std': stdWP1
			}
		
		if sensor_type in [1, 2]:  # SWIR or both
			print (' SWIR or BOTH sensor select: '+str(sensor_type))
			
			FirstBandWP1 = data_masked[:, :, 276]
			SecondBandWP1 = data_masked[:, :, 216]

			water1 = 0
			water1 = np.divide(FirstBandWP1, SecondBandWP1)

			mean_water1 = np.nanmean(water1)
			min_water1 = np.nanmin(water1)
			max_water1 = np.nanmax(water1)
			median_water1 = np.nanmedian(water1)
			std_water1 = np.nanstd(water1)

			# Create a dictionary with the statistics for this image set
			swir_stats = {
				'File Name': dataFileNameWithoutExt,
				'Label': label,
				'NDWI Mean': mean_water1,
				'NDWI Min': min_water1,
				'NDWI Max': max_water1,
				'NDWI Median': median_water1,
				'NDWI Std': std_water1
			}

		if sensor_type == 2:
			print ('BOTH sensor select: '+str(sensor_type))
			
			combined_stats.update(vnir_stats)
			combined_stats.update(swir_stats)


		# Choose the appropriate stats dictionary to append to resultsList
		##########################################################################################
		#                                                                                        #
		#                                     Custom Indice                                      #
		#                                                                                        #
		##########################################################################################

		stats_to_use = vnir_stats if sensor_type == 0 else swir_stats if sensor_type == 1 else combined_stats

		# If custom indices are defined, update stats_to_use accordingly
		for custom_def in custom_index_list:
			print("Processing custom_def:", custom_def)
			custom_index, index_name = calculate_custom_index(
				data_masked,
				custom_def["formula"],
				custom_def["bands"],
				custom_def["name"]
			)
			print("custom_index:", type(custom_index), "index_name:", index_name)

			if custom_index is not None:
				stats = {
					f"{index_name} mean": np.nanmean(custom_index),
					f"{index_name} min": np.nanmin(custom_index),
					f"{index_name} max": np.nanmax(custom_index),
					f"{index_name} median": np.nanmedian(custom_index),
					f"{index_name} std": np.nanstd(custom_index)
				}
				stats_to_use.update(stats)

		resultsList.append(stats_to_use)

		# Export results
		current_results_df = pd.DataFrame([stats_to_use])
		results_df = pd.concat([results_df,current_results_df])
		export_results(current_results_df, outputFileName, output_format, setIndex)

	progress_text_label.config(text=f"Done: {numSets}/{numSets} images processed.")

	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))
	print ('IWASHEREATTHEEND'+str(sensor_type))

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

	print ('ALLGOODUNTIL'+str(sensor_type))
	print ('ALLGOODUNTIL'+str(sensor_type))
	print ('ALLGOODUNTIL'+str(sensor_type))
	print ('ALLGOODUNTIL'+str(sensor_type))
	print ('ALLGOODUNTIL'+str(sensor_type))
	
	print ('wavelengths'+str(wavelengths))
	print ('mean_spectrum'+str(mean_spectrum))
	
	
	#BU YOLLADIKDAN SONRADIR
	#BU YOLLADIKDAN SONRADIR
	#BU YOLLADIKDAN SONRADIR
	# Plot the mean spectra of each set in gray
	# 1️⃣ **Always plot individual mean spectra per image**
	plt.figure(figsize=(10, 6))

	if labels_exist:
		# 1️⃣ Plot individual spectra in gray
		for image_name in mean_spectra_dict:
			if image_name.startswith("Image"):
				for mean_spectrum in mean_spectra_dict[image_name]:
					sns.lineplot(x=wavelengths, y=mean_spectrum, color='gray', alpha=0.4, linestyle='dotted')

		# 2️⃣ Plot average spectrum per class in color with labels
		for label in sorted(set(labels)):
			spectra = mean_spectra_dict.get(label, [])
			if spectra:
				mean_spectrum = np.mean(spectra, axis=0)
				sns.lineplot(x=wavelengths, y=mean_spectrum, label=label, linewidth=2.5)

		plt.legend(title='Class')  # ✅ Legend only for classes

	else:
		# 3️⃣ No labels file: plot all spectra in unique colors, no legend
		for image_name in mean_spectra_dict:
			for mean_spectrum in mean_spectra_dict[image_name]:
				sns.lineplot(x=wavelengths, y=mean_spectrum, linewidth=1)

		# ❌ Explicitly remove any legend if accidentally created
		legend = plt.gca().get_legend()
		if legend:
			legend.remove()

	# Final styling
	plt.xlabel('Wavelength')
	plt.ylabel('Reflectance')
	plt.grid(True)
	plt.tight_layout()

	print(f'All statistics calculated and completed! Results saved to {outputFileName}.')
	print(f'All statistics calculated and completed! Results saved to {outputFileName}.')

	now = datetime.datetime.now()
	print ("Current date and time : ")
	print (now.strftime("%Y-%m-%d %H:%M:%S"))
		
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
	print(f'Statistics calculated and completed!')
	labelGUI.config(text='Analyser finished.',bg='lightgreen')
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
				print("CSV file loaded: "+ file_path)

		except Exception as e:
			#status_label.config(text=f"Error: {str(e)}")
			print("Error:" +str(e))
	

	
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

	# === 1. Save all image spectra to mean_spectra.csv ===
	all_image_spectra = []

	for image_name in sorted(mean_spectra_dict.keys()):
		if image_name.startswith("Image"):
			mean_spectrum = mean_spectra_dict[image_name][0]  # One per image
			all_image_spectra.append([image_name] + list(mean_spectrum))

	# Write to a single file
	with open("mean_spectra.csv", mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Image"] + list(wavelengths))  # header
		writer.writerows(all_image_spectra)

	# === 2. Save per-class spectra if labels file is used ===
	if labels_exist:
		class_data = {label: [] for label in set(labels)}
		
		for label in set(labels):
			for i, image_label in enumerate(labels):
				if label == image_label:
					spectrum = mean_spectra_list[i]
					class_data[label].append(["Image " + str(i + 1)] + list(spectrum))

		for label, spectra_rows in class_data.items():
			filename = f"mean_spectra_{label}.csv"
			with open(filename, mode='w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(["Image"] + list(wavelengths))
				writer.writerows(spectra_rows)
	
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
	print(f'All statistics calculated and completed! Results saved to {outputFileName}.')
	print(f'SIRI')
		
	#NEW Show the plot
	#plt.show()	


def main():

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
				print('secildim')
				folderingo = filedialog.askopenfile(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select your model:",
                                    filetypes=my_filetypes)
				print (folderingo)
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
				print (selected)
				folderingo = './Models/'+selected
				print (folderingo)
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

	folder = './Models'
	filelist = [fname[:-3] for fname in os.listdir(folder) if fname.endswith('.h5')]

	if len(filelist)==0:
		print ('nofileinmodelsfolder')
	


		
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
	
	lRGB = tk.Label(middle, bg='white', width=200, text='RGB Image Slices\n(preview of 5)',padx=10, pady=10, justify='center')
	lRGB.pack()
	
	lMask = tk.Label(right, bg='white', width=200, text='Mask Images\n(preview of 5)',padx=10, pady=10, justify='center')
	lMask.pack()
	
	#l2 = tk.Label(tab1, bg='white', width=200, text='Segmentor\nPlease select which model type you want to use for segmentation:',padx=10, pady=10, justify='left')
	l2 = tk.Label(left, bg='white', width=200, text='Segmentor\nSelect Model:',padx=10, pady=10, justify='center')
	#l2.grid(row=1,column=0)
	l2.pack()
	
	#progressbar = ttk.Progressbar(mode="indeterminate")
	progressbar = ttk.Progressbar(left)
	
	#def CallSegmentation(infold,outfold):
	def CallSegmentationExtra(infold,outfold,modelselection,label):
		global forSingleFile
		#os.system('python segmentor.py')
		#input_dir = "RGB Images"
		label.config(text='Segmentor started...', bg='red')
		
		progressbar.pack()
		progressbar.start()
		progressbar.step(3)
		application_window.update()
		application_window.update_idletasks()

			
		input_dir = infold
		#output_dir = "Masks"
		output_dir = outfold
		
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		output_dir ='./Masks-'+now.strftime("%Y%m%d_%H%M%S")
		forSingleFile = output_dir
		
		#output_dir ='Masks-'+now.strftime("%Y%m%d_%H%M%S")
		print ('hoppax:'+str(outfold))
		print ('hoppa2:'+str(output_dir))
		#model_path = "./Models/Model_UCD_UPJV_VNIR_EPOCH50.h5"
		model_path = modelselection

		print('model:'+str(model_path))
		# Create the output directory
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		# Image Segmentation
		segmenter = Segmentor(input_dir, output_dir, model_path, progressbar, left)
		segmenter.run_segmentation()
		label.config(text='Segmentor finished.\nFiles are ready in '+output_dir[2:]+' folder.\nPlease select which model type you want to (re)start the segmentation:',bg='lightgreen')
		
		progressbar.stop()
		#os.startfile(answer2[2:])
		os.startfile(output_dir[2:])
		#print(input_dir)
		#print(output_dir[2:])
		
		Bsingle["state"] = "normal"
		
		if not os.path.exists(output_dir[2:]+'/originals'):
			os.makedirs(output_dir[2:]+'/originals')
		
		#Maskframe.delete()
		for widgets in Maskframe.winfo_children():
			widgets.destroy()
		
		spaceforImagesMasks = tk.Label(Maskframe,text='')
		spaceforImagesMasks.pack()
		spaceforImagesMasks2 = tk.Label(Maskframe,text='')
		spaceforImagesMasks2.pack()
		spaceforImagesMasks3 = tk.Label(Maskframe,text='')
		spaceforImagesMasks3.pack()
		
		count=0
		for (root_, dirs, files) in os.walk(output_dir):
				if files:
					for file_ in files:
						print(file_[-3:])
						print(file_)
						if(file_[-3:]!='png'):
							pass
						elif '-PlantMask' in file_:						
						#else:
							shutil.copy(input_dir+'/'+file_.replace('-PlantMask', ''), output_dir[2:]+'/originals')
							count=count+1
							if count<6:
								#break
								photo_name = tk.Label(Maskframe,text='IMAGE #'+str(count)+': '+file_)
								path = os.path.join(output_dir, file_)
								image_ = PIL.Image.open(path)
								n_image = image_.resize((500, 500))
								photo = PIL.ImageTk.PhotoImage(n_image)
								photo_name.pack()
								img_label = tk.Label(Maskframe, image=photo)
								img_label.photo = photo                             # <--
								img_label.pack()
								#shutil.copy(input_dir+'/'+file_.replace('-PlantMask', ''), output_dir[2:]+'/originals')
						else:
							pass


		
				
					
	def CallSegmentationInit():

		CallSegmentationExtra2()
		
	def CallSegmentationExtra2():
		
		yarro = './Models/'+optmenu.get()+'.h5'
		print ('optmenu get:'+optmenu.get())
		if optmenu.get() != 'IMPORT MY MODEL' and optmenu.get() != 'Please Select Model':
			print('burayada geldim')
			yarro = folderingo+'.h5'
		elif importButtonCheck==True:
			yarro = folderingo.name
		else:	
			yarro = folderingo.name
			
		print('answer'+answer)
		print('answer2'+answer2)
		print(yarro)
		
		
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
	
	print (optmenu.get())
	#yarro = './Models/'+str(selectedCity)

	


	
	print (optmenu.get())
	print (optmenu.get())
	print (optmenu.get())
	print (optmenu.get())
	
	
	def CallConventor():
		lconv.config(text='I am tring to run...')
		
	def CallHDRCreator():

		lhdr.config(text='HDR Creator\nRunning (asking for input)...',bg='red')
		l.config(bg='white')
		l2.config(bg='white')
		#B["state"] = "disable"
		#B2["state"] = "disable"
		
		print('HDR Creator tring to run...')
		
		#B["state"] = "normal"
		#B2["state"] = "normal"
		


		samples = simpledialog.askinteger("Samples:", "Enter the number of samples:",
										 parent=application_window,
										 minvalue=0, maxvalue=5000)
		if samples is not None:
			print("Samples: ", samples)
		else:
			print("You don't have samples?")
			
		lines = simpledialog.askinteger("lines", "Enter the number of lines:",
										 parent=application_window,
										 minvalue=0, maxvalue=5000)
		if lines is not None:
			print("Lines: ", lines)
		else:
			print("You don't have lines?")
			
		bands = simpledialog.askinteger("Bands:", "Enter the number of bands:",
										 parent=application_window,
										 minvalue=0, maxvalue=5000)
		if bands is not None:
			print("Bands: ", bands)
		else:
			print("You don't have bands?")
			
		bits = simpledialog.askinteger("Data Type Bits:", "Enter the data type (bits):",
										 parent=application_window,
										 minvalue=0, maxvalue=5000)
		if bits is not None:
			print("Bits: ", bits)
		else:
			print("You don't have bits?")
			
		byteorder = simpledialog.askinteger("Byte Order:", "Enter the byte order: ",
										 parent=application_window,
										 minvalue=0, maxvalue=5000)
		if byteorder is not None:
			print("Byteorder: ", byteorder)
		else:
			print("You don't have byteorder?")

		starting_wavelength = simpledialog.askfloat("Starting Wavelength:", "Enter the starting wavelength:",
									   parent=application_window,
									   minvalue=0.0, maxvalue=5000.00)
		if starting_wavelength is not None:
			print("Starting wavelength:", starting_wavelength)
		else:
			print("You don't have a starting wavelength?")
			
		ending_wavelength = simpledialog.askfloat("Ending Wavelength:", "Enter the ending wavelength: ",
									   parent=application_window,
									   minvalue=0.0, maxvalue=5000.00)
		if ending_wavelength is not None:
			print("Ending wavelength", ending_wavelength)
		else:
			print("You don't have a ending wavelength?")
			
			
		file_name = simpledialog.askstring("File Name:", "Enter the file name:",
                                parent=application_window)
		if file_name is not None:
			print("Your file name is ", file_name)
		else:
			print("You don't have a file name?")
		
	
    
    

		lhdr.config(text='HDR Creator\nRunning (creating files based on input)...',bg='red')
		print ('HDR Creator started.')
		CallHDRCreatorExtra(int(samples), int(lines), int(bands), int(bits), int(byteorder), (float(starting_wavelength), float(ending_wavelength)), file_name,lhdr)
		


	def CallSegmentation():
		global answer
		
		global RGBCheck
		global ModelCheck
		

		l2.config(text='RGB Image Slices are ready...',)
		#B2["state"] = "disable"
		#B4["state"] = "disable"
		lhdr.config(bg='white')
		l.config(bg='white')
		
		# Build a list of tuples for each file type the file dialog should display
		my_filetypes = [('all files', '.*'), ('text files', '.txt')]

		#os.chdir("..")
		# Ask the user to select a folder.
		answer = filedialog.askdirectory(parent=application_window,
									 initialdir=os.getcwd(),
									 title="Please select your input folder (for RGB Images):")

		#os.chdir("..")
		# Ask the user to select a folder.
		
		#THIS IS EDITED FOR THE FLOW OF THE APP
		
		#answer2 = filedialog.askdirectory(parent=application_window,
		#							 initialdir=os.getcwd(),
		#							 title="Please select your output folder (for Mask files):")
		#answer2 = 'no'
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		answer2 ='./Masks-'+now.strftime("%Y%m%d_%H%M%S")
		
		#print ('./Models/'+optmenu.get())
		
		yarro = './Models/'+optmenu.get()+'.h5'
		
		
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
					print(file_[-3:])
					print(file_)
					if(file_[-3:]!='png'):
						pass
					else:
						count=count+1
						if count>5:
							break
						photo_name = tk.Label(RGBframe,text='IMAGE #'+str(count)+': '+file_)
						path = os.path.join(answer, file_)
						image_ = PIL.Image.open(path)
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
				print(answer+"/"+images)
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
		if answer is None:
					RGBCheck=False
					Bstart["state"] = "disabled"
					Bfake = tk.Button(tab3,text="")
					orig_color = Bfake.cget("bg")
					Bstart.configure(bg=orig_color,fg='black')
					l2.config(text='Segmentor\n\nPlease select RGB Slices folder.',fg='red')
		if answer=='':
					RGBCheck=False
					Bstart["state"] = "disabled"
					Bfake = tk.Button(tab3,text="")
					orig_color = Bfake.cget("bg")
					Bstart.configure(bg=orig_color,fg='black')
					l2.config(text='Segmentor\n\nPlease select RGB Slices folder.',fg='red')
		
		
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
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
			
		# Define the filename for the output file (CSV or XLSX)
		outputFileName2 = 'Results-'+now.strftime("%Y%m%d_%H%M%S")+'.xlsx'  # Change the filename and extension as needed
		
		'''		
		# Export the results DataFrame to CSV or XLSX
		if outputFileName2.endswith('.csv'):
			results_df.to_csv(ResFol+'/'+outputFileName2, index=False)
		elif outputFileName2.endswith('.xlsx'):
			results_df.to_excel(ResFol+'/'+outputFileName2, sheet_name='Analysis results', index=False)
		else:
			raise ValueError('Unsupported file format. Use .csv or .xlsx')

		l.config(text='Your results are ready in\n'+outputFileName2+'\n at '+ResFol,bg='lightgreen')
		os.startfile(ResFol)
		#os.startfile(os.getcwd())
		#os.startfile(os.getcwd()+'/'+outputFileName2)
		'''
		#export_results(current_results_df, ResFol+'/'+outputFileName2, 'xlsx', 0)
		export_results(results_df, ResFol+'/'+outputFileName2, 'xlsx', 0)
		
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
		os.startfile(ResFol)
		#os.startfile(os.getcwd())
		#os.startfile(os.getcwd()+'/'+outputFileName)
		'''
		#export_results(current_results_df, ResFol+'/'+outputFileName, 'csv', 0)
		export_results(results_df, ResFol+'/'+outputFileName, 'csv', 0)
	
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

	
	def BrowseForExportImg():
		global ResFol2
		ResFol2 = filedialog.askdirectory(parent=application_window,
					initialdir=os.getcwd(),
					title="Please select the folder to save Image and Mean Spectra files:")
					
		if ResFol2 != '':
			LResFol2.config(text='Folder: '+ResFol2,bg='white',fg='black')
			BIexport1["state"]="normal"
			#BIexport2["state"]="normal"
			BIexportImg["state"]="normal"
			#BIexportImg2["state"]="normal"
		else:
			LResFol2.config(text='Please pick a folder to save Image and Mean Spectra files.',bg='white',fg='black')
			BIexport1["state"] = "disabled"
			#BIexport2["state"] = "disabled"
			BIexportImg["state"]="disabled"
			#BIexportImg2["state"]="disabled"
	
	def ExportCSVMean():
		global Resfol2
		global mean_spectra_list
		global wavelengths
		global mean_spectra_dict
		print ('I am here csv')
			# Export the mean spectra values for each BIL file
			
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

		
		with open(ResFol2+'/'+'mean_spectra-'+now.strftime("%Y%m%d_%H%M%S")+'.csv', mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['Wavelength'] + [f'Mean_Spectrum_{i+1}' for i in range(len(mean_spectra_list))])
			for i in range(len(wavelengths)):
				row = [wavelengths[i]] + [mean_spectra_list[j][i] for j in range(len(mean_spectra_list))]
				writer.writerow(row)


		#if labels_exist:
		if useorno.get() and answerLabels != '':
			# Export the mean spectra values per label
			for label in mean_spectra_dict.keys():
				output_filename = f'mean_spectra_{label}-'+now.strftime("%Y%m%d_%H%M%S")+'.csv'
				with open(ResFol2+'/'+output_filename, mode='w', newline='') as file:
					writer = csv.writer(file)
					writer.writerow(['Wavelength'] + [f'Mean_Spectrum_{label}_{i+1}' for i in range(len(mean_spectra_dict[label]))])
					for i in range(len(wavelengths)):
						row = [wavelengths[i]] + [mean_spectra_dict[label][j][i] for j in range(len(mean_spectra_dict[label]))]
						writer.writerow(row)

				
		os.startfile(ResFol2)
		
	def ExportXLSXMean():
		global Resfol2
		global mean_spectra_list
		global wavelengths
		print ('I am here xlsx')

		
			# Export the mean spectra values for each BIL file
			
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

		
		with open(ResFol2+'/'+'mean_spectra-'+now.strftime("%Y%m%d_%H%M%S")+'.xlsx', mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['Wavelength'] + [f'Mean_Spectrum_{i+1}' for i in range(len(mean_spectra_list))])
			for i in range(len(wavelengths)):
				row = [wavelengths[i]] + [mean_spectra_list[j][i] for j in range(len(mean_spectra_list))]
				writer.writerow(row)


		#if labels_exist:
		if useorno.get() and answerLabels != '':
			# Export the mean spectra values per label
			for label in mean_spectra_dict.keys():
				output_filename = f'mean_spectra_{label}-'+now.strftime("%Y%m%d_%H%M%S")+'.xlsx'
				with open(ResFol2+'/'+output_filename, mode='w', newline='') as file:
					writer = csv.writer(file)
					writer.writerow(['Wavelength'] + [f'Mean_Spectrum_{label}_{i+1}' for i in range(len(mean_spectra_dict[label]))])
					for i in range(len(wavelengths)):
						row = [wavelengths[i]] + [mean_spectra_dict[label][j][i] for j in range(len(mean_spectra_dict[label]))]
						writer.writerow(row)		
		
		os.startfile(ResFol2)
	
	def ExportIMG():
		global load
		print ('I am here image')
		
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
		
		load.save(ResFol2+'/'+'ResultsImage-'+now.strftime("%Y%m%d_%H%M%S")+'.png', 'PNG')

		ExportIMGJPG()
		
		os.startfile(ResFol2)

	def ExportIMGJPG():
		global load
		print ('I am here image')
		
		now = datetime.datetime.now()
		print ("Current date and time : ")
		print (now.strftime("%Y-%m-%d %H:%M:%S"))
		#timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
		
		rgb_load = load.convert('RGB')
		rgb_load.save(ResFol2+'/'+'ResultsImage-'+now.strftime("%Y%m%d_%H%M%S")+'.jpg')
		os.startfile(ResFol2)
		
	def JumpToA():
		tabControl.select(tab4)	
	
	def JumpToU():
		tabControl.select(tab3)	

	def JumpToV():
		tabControl.select(tab5)			
		
	
	def Dark():
		global answer3
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('hdr files', '.hdr'),('all files', '.*'), ('text files', '.txt')]

		# Ask the user to select a single file name.
		answer3 = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a Dark HDR file:",
                                    filetypes=my_filetypes)
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
		answer3b = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a Data HDR file:",
                                    filetypes=my_filetypes)
		
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
		answer3c = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a White HDR file:",
                                    filetypes=my_filetypes)
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
		global DarkCheck
		global DataCheck
		global BilCheck
		global WhiteCheck
		global MaskCheck
		
		my_filetypes = [('BIL files', '.bil'),('RAW files', '.raw'),('all files', '.*'), ('text files', '.txt')]
		answer4 = filedialog.askdirectory(parent=application_window,
                                 initialdir=os.getcwd(),
                                 title="Please select your input folder (for BIL files):")
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
		answer5 = filedialog.askdirectory(parent=application_window,
									 initialdir=os.getcwd(),
									 title="Please select your Masks folder (Segmentor output folder):")
	
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
			
	def CallAnalyser():
		global user_index_name
		global user_formula
		global user_bands
		l.config(text='Analyser running...',bg='red')
		lhdr.config(bg='white')
		l2.config(bg='white')
		
		#B["state"] = "disable"
		#B4["state"] = "disable"
		
		my_filetypes = [('all files', '.*'), ('text files', '.txt')]

		#print ('Called Analyser'+var.get())
		#yarro = './Models/'+optmenu.get()
		#CallSegmentationExtra(answer,answer2,yarro)
			
		#os.chdir("..")
		# Ask the user to select a folder.
		
		# Ask the user to select a single file name.
		#answer3 = filedialog.askopenfilename(parent=application_window,
                                    #initialdir=os.getcwd(),
                                    #title="Please select a Dark HDR file:",
                                    #filetypes=my_filetypes)
									
		# Ask the user to select a single file name.
		#answer3b = filedialog.askopenfilename(parent=application_window,
                                    #initialdir=os.getcwd(),
                                    #title="Please select a Data HDR file:",
                                    #filetypes=my_filetypes)
		
		# Ask the user to select a single file name.
		#answer3c = filedialog.askopenfilename(parent=application_window,
                                    #initialdir=os.getcwd(),
                                    #title="Please select a White HDR file:",
                                    #filetypes=my_filetypes)
	

		#if var.get()=='True':
		#answer3 = filedialog.askdirectory(parent=application_window,
        #                         initialdir=os.getcwd(),
        #                         title="Please select your input folder (for ENVI files):")
		#else:
		#	answer3 = filedialog.askdirectory(parent=application_window,
        #                         initialdir=os.getcwd(),
        #                         title="Please select your input folder (for HDR files):")
	
	
		#answer4 = filedialog.askdirectory(parent=application_window,
                                 #initialdir=os.getcwd(),
                                 #title="Please select your input folder (for BIL files):")
								 
		#print ('./Models/'+var.get())
		
		l.config(text='Analyser running...',bg='red')
		
		#answer5 = filedialog.askdirectory(parent=application_window,
									 #initialdir=os.getcwd(),
									 #title="Please select your Masks folder (Segmentor output folder):")
		 # Mapping of band names to indices (I am using NDVI as the example even though I am already including it :-) )
		
		CallAnalyserExtra(answer3,answer3b,answer3c,answer4,answer5,l,tab4,tree,tabControl,tab5,B2jump4,Resultsframe,resultsShow,Bexport1,Bexport2,Bexport3,LResFol,progressbar2,BIexport3)
		#l.config(text='Analyser finished.')
		#B["state"] = "normal"
		#B4["state"] = "normal"
		#print ('bittimi1')
		#return


	#Bfolder=tk.Button(tab1,text="Select Folder with PNG Files for Segmentation",command= CallSegmentation )
	#Bstart=tk.Button(tab1,text="Start Segmentor",command= CallSegmentation )
	#Bfolder=tk.Button(left,text="Select Folder with PNG Files for Segmentation",command= CallSegmentation )
	#Bstart=tk.Button(left,text="Start Segmentor",command= CallSegmentation )
	def Model1():
		print('Model1')
		global folderingo
		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
			
		folderingo = './Models/DICOT'
		print (folderingo)
		l2.config(text='Segmentor\n\nModel Selected.')
		
		

		return folderingo
		
	def Model2():
		print('Model2')
		global folderingo
		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="lightgreen",fg='white')
		
		folderingo = './Models/GYMNOSPERM'
		print (folderingo)
		l2.config(text='Segmentor\n\nModel Selected.')
		
		return folderingo

	def Model3():	
		print('Model3')
		global folderingo

		global RGBCheck
		global ModelCheck
		
		ModelCheck=True
		importButtonCheck=False
		
		if(RGBCheck):
			Bstart["state"] = "normal"
			#Bsingle["state"] = "normal"
			Bstart.configure(bg="green",fg='white')
			
		folderingo = './Models/MONOCOT'
		print (folderingo)
		l2.config(text='Segmentor\n\nModel Selected.')
		
		return folderingo

	def Model4():	
		print('Model4')
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
		
		print('secildim')
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
	def edit_selection():
		global sensorselect
		#global answerLabels
		if sensorselect.get() == 0:
			B2bil['text']='Load BIL/RAW Files'
		elif sensorselect.get() == 1:
			B2bil['text']='Load SWIR/RAW Files'
		elif sensorselect.get() == 2:
			B2bil['text']='Load BIL/SWIR/RAW Files'
			
	def SingleView():
		global forSingleFile
		print(forSingleFile+'\originals')
		forSingleFileFolder=forSingleFile+'\originals'
		
		my_filetypes = [('PNG files', '.png'),('all files', '.*'), ('text files', '.txt')]
		answerSingle = filedialog.askopenfilename(parent=application_window,
                                    initialdir=forSingleFileFolder,
                                    title="Please select an RGB Slice\n(from originals folder within any Masks folder):",
                                    filetypes=my_filetypes)
									

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
		print('deneme2: '+deneme)
		photo_name = tk.Label(RGBframe,text='IMAGE #1: '+deneme)
		#path = os.path.join(output_dir, file_)
		image_ = PIL.Image.open(answerSingle)
		n_image = image_.resize((500, 500))
		photo = PIL.ImageTk.PhotoImage(n_image)
		photo_name.pack()
		img_label = tk.Label(RGBframe, image=photo)
		img_label.photo = photo                             # <--
		img_label.pack()

		deneme = answerSingle.replace('originals/','')
		answerSingleMask = deneme[:-4]+'-PlantMask.png'
		
		#deneme  = answerSingle[:-4]+'-PlantMask.png'.replace('originals/','')
		print(deneme)
		print(answerSingleMask)
		buseferoldu=answerSingleMask.replace(str(mainfolder),'').replace(str(filenamegibi[:-9]).replace('originals/',''),'')
		print(buseferoldu)
		photo_name = tk.Label(Maskframe,text='IMAGE #1: '+buseferoldu)
		#path = os.path.join(output_dir, file_)
		image_ = PIL.Image.open(answerSingleMask)
		n_image = image_.resize((500, 500))
		photo = PIL.ImageTk.PhotoImage(n_image)
		photo_name.pack()
		img_label = tk.Label(Maskframe, image=photo)
		img_label.photo = photo                             # <--
		img_label.pack()
	
	print('noluyor acaba' + str(len(filelist)))
	Bfolder=tk.Button(left,text="Load RGB Slices",command= CallSegmentation )
	
	Bsingle=tk.Button(left,text="Preview Individual Image and Mask",command= SingleView )
	
	now = datetime.datetime.now()
	print ("Current date and time : ")
	print (now.strftime("%Y-%m-%d %H:%M:%S"))
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
	
	lheader = tk.Label(tab3, bg='white', width=200, text='Load Header Files',padx=10, pady=10)
	lheader.pack()
	
	B2dark=tk.Button(tab3,text="DARK",command= Dark )
	B2data=tk.Button(tab3,text="DATA",command= Data )
	B2white=tk.Button(tab3,text="WHITE",command= White )
	B2dark.pack()
	B2data.pack()
	B2white.pack()
	
	lheader2 = tk.Label(tab3, bg='white', width=200, text='',padx=10, pady=10)
	lheader2.pack()
	
	checkSensors = tk.Label(tab3,text='Select your sensor type:')
	checkSensors.pack(anchor='center')
	
	#vars_list.append(var)
	# 0 for VNIR, 1 for SWIR, 2 for both
	#tk.Radiobutton(tab3, text="VNIR", variable=sensorselect, value=0).pack(anchor='center',side="left")
	#tk.Radiobutton(tab3, text="SWIR", variable=sensorselect, value=1).pack(anchor='center',side="left")
	#tk.Radiobutton(tab3, text="Both", variable=sensorselect, value=2).pack(anchor='center',side="left")
	#tk.Radiobutton(tab3, text="VNIR", variable=sensorselect, value=0,command=edit_selection).pack(anchor='center')
	#tk.Radiobutton(tab3, text="SWIR", variable=sensorselect, value=1,command=edit_selection).pack(anchor='center')
	#tk.Radiobutton(tab3, text="Both", variable=sensorselect, value=2,command=edit_selection).pack(anchor='center')
	tk.Radiobutton(tab3, text="VNIR", variable=sensorselect, value=0).pack(anchor='center')
	tk.Radiobutton(tab3, text="SWIR", variable=sensorselect, value=1).pack(anchor='center')
	tk.Radiobutton(tab3, text="Both", variable=sensorselect, value=2).pack(anchor='center')
	#tk.Radiobutton(tab3, text="Yes", variable=sensorselect, value=True,command=print_selection).pack(anchor='center')
	#tk.Radiobutton(tab3, text="No", variable=sensorselect, value=False,command=print_selection).pack(anchor='center')
	
	
	B2bil=tk.Button(tab3,text="Load BIL/RAW Files",command= Bil )
	B2mask=tk.Button(tab3,text="Load Segmentation Masks",command= Mask )
	
	#tk.Label(tab3, text="Index Name: ").grid(row=0)
	#tk.Label(tab3, text="Var1: ").grid(row=1)
	#tk.Label(tab3, text="Var2: ").grid(row=2)
	#tk.Label(tab3, text="Var3: ").grid(row=3)
	#tk.Label(tab3, text="Formula: ").grid(row=4)
	
	formulaframe = Frame(tab3)
		
	e1 = tk.Entry(formulaframe, width=7)
	e2 = tk.Entry(formulaframe, width=7)
	#e3 = tk.Entry(formulaframe)
	e4 = tk.Entry(formulaframe, width=35)
	e4.insert(0, '(Var1 - Var2) / (Var1 + Var2)')
	e5 = tk.Entry(formulaframe, width=15)

	
	lheaderx = tk.Label(formulaframe, bg='white', text='- User Defined Index & Formula -\n\nPlease give a name for your index, values for your variables\nand use them as Var1 & Var2 in your Formula\n(leave Index Name empty if you do not want to use custom formula)',padx=10, pady=10).pack(side = LEFT)
	#lheaderx
	#e5.grid(row=0, column=1)
	#e1.grid(row=1, column=1)
	#e2.grid(row=2, column=1)
	#e3.grid(row=3, column=1)
	#e4.grid(row=4, column=1)
	
	bottomframe = Frame(tab3)
	bottomframe.pack( side = BOTTOM )
	

	

	B2bil.pack()
	B2mask.pack()
	
	lheader3 = tk.Label(tab3, bg='white', width=200, text='',padx=10, pady=10)
	lheader3.pack()
	
	
	B2jump=tk.Button(bottomframe,text="Switch to Analyser",command= JumpToA )
	
	B2jump2=tk.Button(tab5,text="Switch Back to Analyser",command= JumpToA )
	B2jump2.pack()
	B2jump3=tk.Button(tab5,text="Switch Back to Upload Files",command= JumpToU )
	B2jump3.pack()
	
	BIexport3=tk.Button(tab5,text="Browse...",command= BrowseForExportImg )	
	LResFol2 = tk.Label(tab5, text="")
	BIexport1=tk.Button(tab5,text="Export Mean Spectra files",command= ExportCSVMean )
	BIexport2=tk.Button(tab5,text="Export Mean Spectra files as XLSX",command= ExportXLSXMean )
	BIexportImg=tk.Button(tab5,text="Save Image",command= ExportIMG )
	BIexportImg2=tk.Button(tab5,text="Save Image as JPG",command= ExportIMGJPG )
	
	resultsShow = Text(tab5, state='disable')
	
	resultsShow.pack(fill='both', expand=True)
	Resultsframe = tk.Frame(resultsShow)
	resultsShow.window_create("end", window=Resultsframe)
	
	
	B2jump4=tk.Button(tab4,text="Check the Spectra Visualiser",command= JumpToV )
	
	
	
	B2=tk.Button(tab4,text="Start Analyser",command= CallAnalyser )
	progressbar2 = ttk.Progressbar(tab4)
	
	tree = ttk.Treeview(tab4, show="headings")
	
	
	Bexport3=tk.Button(tab4,text="Browse...",command= BrowseForExport )	
	LResFol = tk.Label(tab4, text="")
	Bexport1=tk.Button(tab4,text="Export as CSV File",command= ExportCSV )
	Bexport2=tk.Button(tab4,text="Export as XLSX File",command= ExportXLSX )
	
	
	
	
	
	
	#B2=tk.Button(application_window,text="Start Analyser",command= CallAnalyser )
	
	 
	def print_selection():
		global useorno
		global answerLabels
		if useorno.get():
			lheader3.config(text='You have selected to use Labels File')
			
			my_filetypes = [('CSV files', '.csv'),('all files', '.*'), ('text files', '.txt')]
			answerLabels = filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select the Labels file:",
                                    filetypes=my_filetypes)
			if answerLabels == '':
				lheader3.config(text='You DID NOT select a Labels file, you can still continue to Analyser without one.')
			else:
				lheader3.config(text='You have selected to use '+ answerLabels)	
		else:
			lheader3.config(text='You have selected NOT to use Labels File')
		#account.config(text='This is possible.' + str(useorno.get()) + ' Files')
		#B2.pack()
		
	#lconv = tk.Label(application_window, bg='white', width=200, text='Convert HDR to ENVI',padx=10, pady=10)
	#lconv.pack()
	#B3=tk.Button(application_window,text="Start HDRtoENVI Convertor",command= CallConventor )
	#B3.pack();

	# function to validate mark entry
	def only_numbers(char):

		#if "." in char:
		#	return False
		#else:
		#	return char.isdigit() or char=='.'
		
		return char.isdigit()

	def only_decimals(char):

		#if "." in char:
		#	return False
		#else:
		#	return char.isdigit() or char=='.'	
			
		return char.isdigit() or char=='.'		

	
	def only_1or0(char,action):
		global yahu
		print('lan:' + char)
		print('act:' + action)
		print('yahu:' + yahu.get())		
		
		if (action=='0'):
			#yahu.set('')
			#Eby.delete(0, END)
			return True
		#if len(Eby.get())>1:
			#print(len(Eby.get()))
		#Eby.delete(1,'end')
			#Eby.keyset(Eby.get()[-1])
			#Eby.delete(1,END)
			#.set(entry_text.get()[-1])
		elif (char=='0' or char=='1'):
			print(len(Eby.get()))
			if (len(Eby.get())==0):
				print('put: '+ str(len(Eby.get())))
				return True
			else:
				return False
		else:
			print(len(Eby.get()))
			return False
		
	
	def OnKeyRelease(some):
		print(some)
		if len(Lfl2["text"])!= 0 and Es.index("end") != 0 and El.index("end") != 0 and Eba.index("end") != 0 and Ebi.index("end") != 0 and Eby.index("end") != 0 and Esw.index("end") != 0 and Eew.index("end") != 0 and Efn.index("end") != 0:
			print ('bos birakmadin')
			B4["state"] = "normal"
			B4.configure(bg="lightgreen")
		else:	
			Bfake = tk.Button(tab3,text="")
			orig_color = Bfake.cget("bg")
			B4["state"] = "disabled"
			B4.configure(bg=orig_color)		

	#def check(event):

	#	text = event.widget.get()
	#	print('text:', text)

	#	parts = text.split('.')
	#	parts_number = len(parts)

	#	if parts_number > 2:
	#		print('too much dots')
	#		return False
			#event.widget.delete(0,END)
			#event.insert(0,text)
			#event.widget.insert(0,text[:-1])
	#	else:
	#		return True

		#if parts_number > 1 and parts[1]: # don't check empty string
		#	if not parts[1].isdecimal() or len(parts[1]) > 2:
		#		print('wrong second part')

		#if parts_number > 0 and parts[0]: # don't check empty string
		#	if not parts[0].isdecimal() or len(parts[0]) > 8:
		#		print('wrong first part')			
	

	def checkfloat(text):
	
		print('text:', text)

		parts = text.split('.')
		parts_number = len(parts)

		if parts_number > 2:
			print('too much dots')
			#Es.config(bg="red")
			return False
		else:
			return True
			#event.widget.delete(0,END)
			#event.insert(0,text)
			#event.widget.insert(0,text[:-1])
	#	else:
	#		return True

		#if parts_number > 1 and parts[1]: # don't check empty string
		#	if not parts[1].isdecimal() or len(parts[1]) > 2:
		#		print('wrong second part')

		#if parts_number > 0 and parts[0]: # don't check empty string
		#	if not parts[0].isdecimal() or len(parts[0]) > 8:
		#		print('wrong first part')			
		
	validation = application_window.register(only_numbers)
	validation2 = application_window.register(only_1or0)
	validation3 = application_window.register(only_decimals)
	

	Ls = Label(tab2, text="Number of Samples (COLUMNS):").grid(row=1)
	#Ls.pack( side = LEFT)
	Es = Entry(tab2, bd =5,validate="key", validatecommand=(validation, '%S'))
	Es.grid(row=1, column=1)
	Es.bind("<KeyRelease>", OnKeyRelease)

	#Es.pack(side = RIGHT)
	#Es.bind('<KeyRelease>', check)

	
	Ll = Label(tab2, text="Number of Lines (ROWS):").grid(row=2)
	#Ll.pack( side = LEFT)
	El = Entry(tab2, bd =5,validate="key", validatecommand=(validation, '%S'))
	El.grid(row=2, column=1)
	#El.pack(side = RIGHT)
	El.bind("<KeyRelease>", OnKeyRelease)
	
	Lba = Label(tab2, text="Number of Spectral Bands:").grid(row=3)
	#Lba.pack( side = LEFT)
	Eba = Entry(tab2, bd =5,validate="key", validatecommand=(validation, '%S'))
	Eba.grid(row=3, column=1)
	#Eba.pack(side = RIGHT)
	Eba.bind("<KeyRelease>", OnKeyRelease)
	
	Lbi = Label(tab2, text="Bit Depth (Bits per Pixel):").grid(row=4)
	#Lbi.pack( side = LEFT)
	Ebi = Entry(tab2, bd =5,validate="key", validatecommand=(validation, '%S'))
	Ebi.grid(row=4, column=1)
	#Ebi.pack(side = RIGHT)
	Ebi.bind("<KeyRelease>", OnKeyRelease)

	Lby = Label(tab2, text="Byte Order (0 or 1):").grid(row=5)
	#Lby.pack( side = LEFT)
	Eby = Entry(tab2, bd =5,textvariable=yahu, validate="key", validatecommand=(validation2, '%S', '%d'))

	Eby.grid(row=5, column=1)
	Eby.bind("<KeyRelease>", OnKeyRelease)
	#Eby.pack(side = RIGHT)

	Lsw = Label(tab2, text="First (Starting) Wavelength (nm):").grid(row=6)
	#Lsw.pack( side = LEFT)
	Esw = Entry(tab2, bd =5,validate="key", validatecommand=(validation3, '%S'))
	Esw.grid(row=6, column=1)
	Esw.bind("<KeyRelease>", OnKeyRelease)
	#Esw.pack(side = RIGHT)

	Lew = Label(tab2, text="Last (Ending) Wavelength (nm):").grid(row=7)
	#Lew.pack( side = LEFT)
	Eew = Entry(tab2, bd =5,validate="key", validatecommand=(validation3, '%S'))
	Eew.bind("<KeyRelease>", OnKeyRelease)
	Eew.grid(row=7, column=1)
	#Eew.pack(side = RIGHT)
	
	Lfn = Label(tab2, text="File Name:").grid(row=8)
	#Lfn.pack( side = LEFT)
	Efn = Entry(tab2, bd =5)
	Efn.bind("<KeyRelease>", OnKeyRelease)
	Efn.grid(row=8, column=1)
	#Efn.pack(side = RIGHT)
	
	
	
	#int(samples), int(lines), int(bands), int(bits), int(byteorder), (float(starting_wavelength), float(ending_wavelength)), file_name,lhdr)
	
	
	def print_HDRinputs(*args):
		global HDRF
		print(Es.get())
		print(El.get())
		print(Eba.get())
		print(Ebi.get())
		print(Eby.get())
		print(Esw.get())
		print(Eew.get())
		print(Efn.get())
		
		if len(Lfl2["text"])== 0 or Es.index("end") == 0 or El.index("end") == 0 or Eba.index("end") == 0 or Ebi.index("end") == 0 or Eby.index("end") == 0 or Esw.index("end") == 0 or Eew.index("end") == 0 or Efn.index("end") == 0:
			print ('bos birakma')
			lhdr.config(text='Please fill all the setting variables...',bg='red')
		#print(Eew.get())
		else:
			samples = int(Es.get())
			lines = int(El.get())
			bands = int(Eba.get())
			bits = int(Ebi.get())
			byteorder = int(Eby.get())
			if (checkfloat(Esw.get())):
				fark1 = float(Esw.get())
				Esw.config(bg="white")
			else:
				Esw.config(bg="red")
				lhdr.config(text='Please fix the decimal value for First (Starting) Wavelength ',bg='red')
				return False
				
			if (checkfloat(Eew.get())):
				fark2 = float(Eew.get())
				Eew.config(bg="white")
			else:
				Eew.config(bg="red")
				lhdr.config(text='Please fix the decimal value for Last (Ending) Wavelength ...',bg='red')
				return False
				
			
			file_name = Efn.get()
			folder=HDRF
			
			lhdr.config(text='HDR Creator\nRunning (creating files based on input)...',bg='red')
			print ('HDR Creator started.')
			CallHDRCreatorExtra(int(samples), int(lines), int(bands), int(bits), int(byteorder), (fark1, fark2), file_name,folder,lhdr)

	def pick_HDRFolder(*args):
		global HDRF
		global useorno
		HDRF = filedialog.askdirectory(parent=application_window,
					initialdir=os.getcwd(),
					title="Please select the folder to save the HDR file:")
		Lfl2.config(text=HDRF,bg='white',fg='black')
		OnKeyRelease('test')
			
	
	Lfl = tk.Label(tab2, text="Folder to Save:")
	Lfl.grid(row=9)
		
	BHDRFolder=tk.Button(tab2,text="Browse...",command= pick_HDRFolder )
	BHDRFolder.grid(row=9,column=3)
	Lfl2 = tk.Label(tab2, text="")
	Lfl2.grid(row=9,column=1)	

	
	
	
	lhdr = tk.Label(tab2, bg='white', text='HDR Creator (calibration input required)',padx=10, pady=10)
	lhdr.grid(row=0,columnspan=2)
	#B4=tk.Button(tab2,text="Start HDR creator",command= CallHDRCreator ).grid(row=9,column=1)	
	B4=tk.Button(tab2,text="Start HDR creator",command= print_HDRinputs )
	B4["state"] = "disabled"
	B4.grid(row=11,column=1)	
	

	
	
	#B4=tk.Button(application_window,text="Start HDR creator",command= CallHDRCreator )	
	#lhdr = tk.Label(application_window, bg='white', width=200, text='HDR Creator (calibration input required)',padx=10, pady=10)

	#THIS SHOULD BACK IN AT THE END
	#lhdr.pack()
	#B4.pack()
	


	l = tk.Label(tab4, bg='white', width=200, text='Analyser',padx=10, pady=10)
	#l = tk.Label(application_window, bg='white', width=200, text='Analyser',padx=10, pady=10)

	l.pack()

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
	BIexport3["state"] = "disabled"
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
	
	checkLabels = tk.Label(tab3,text='Use Labels file?\n(optional)')
	checkLabels.pack(anchor='center')
	

	#useorno = tk.IntVar(value=0)
	#vars_list.append(var)
	#tk.Radiobutton(tab3, text="Yes", variable=var, value=1).pack(anchor='center',side="left")
	#tk.Radiobutton(tab3, text="No", variable=var, value=0).pack(anchor='center',side="left")
	tk.Radiobutton(tab3, text="Yes", variable=useorno, value=True,command=print_selection).pack(anchor='center')
	tk.Radiobutton(tab3, text="No", variable=useorno, value=False,command=print_selection).pack(anchor='center')

	# Custom formula variables
	IndexName = tk.StringVar()
	CustomFormula = tk.StringVar()
	Var1 = tk.StringVar()
	Var2 = tk.StringVar()

	# Frame for the formula input
	formulaframe = tk.Frame(tab3)
	formulaframe.pack(pady=10)

	lheaderx = tk.Label(
    formulaframe,
    bg='white',
    text='- User Defined Index & Formula -\n\nPlease give a name for your index, values for your variables\nand use them as Var1 & Var2 in your Formula\n(leave Index Name empty if you do not want to use custom formula)',
    padx=10, pady=10, justify='left'
	)
	lheaderx.pack(side=tk.LEFT)

	inputs_frame = tk.Frame(formulaframe)
	inputs_frame.pack(side=tk.LEFT, padx=30)

	# Function to add custom formulas
	def GetFormula():
		index_name = IndexName.get()
		formula = CustomFormula.get()
		var1 = Var1.get()
		var2 = Var2.get()

		if index_name and formula and var1 and var2:
			custom_index_list.append({
				"name": index_name,
				"bands": {"Var1": int(var1), "Var2": int(var2)},
				"formula": formula
			})
			print("Custom index list now:", custom_index_list)
			messagebox.showinfo("Formula Added", f"Custom index '{index_name}' added.")

			# Update the summary label
			formula_summary = "\n".join(f"{f['name']}: {f['formula']}" for f in custom_index_list)
			label_summary.config(text=formula_summary)

			# Clear inputs
			IndexName.set("")
			CustomFormula.set("")
			Var1.set("")
			Var2.set("")
		else:
			messagebox.showerror("Error", "Please fill in all custom index fields.")

	# Input fields
	tk.Label(inputs_frame, text="Index Name: ").pack(side=tk.LEFT)
	tk.Entry(inputs_frame, textvariable=IndexName, width=10).pack(side=tk.LEFT)

	tk.Label(inputs_frame, text="Var1: ").pack(side=tk.LEFT)
	tk.Entry(inputs_frame, textvariable=Var1, width=5).pack(side=tk.LEFT)

	tk.Label(inputs_frame, text="Var2: ").pack(side=tk.LEFT)
	tk.Entry(inputs_frame, textvariable=Var2, width=5).pack(side=tk.LEFT)

	tk.Label(inputs_frame, text="Formula:").pack(side=tk.LEFT)
	tk.Entry(inputs_frame, textvariable=CustomFormula, width=25).pack(side=tk.LEFT)

	tk.Button(inputs_frame, text='Add Formula', command=GetFormula).pack(side=tk.LEFT)

	# Frame for displaying formula summary
	summary_frame = tk.Frame(formulaframe)
	summary_frame.pack(fill="x", padx=10, pady=5)

	# Summary label (must be created before GetFormula is ever called)
	label_summary = tk.Label(summary_frame, text="", justify="left", anchor="w", font=("Arial", 10))
	label_summary.pack(pady=(5, 0), anchor="w")

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

	LogoImage = PIL.Image.open("BSA_logo.png")	
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
	show_about_page()	
	application_window.mainloop()
	
	

	#application_window.mainloop()	
	
if __name__ == "__main__":
	if not has_user_agreed():
		show_splash()
		
	main()
	




	







	