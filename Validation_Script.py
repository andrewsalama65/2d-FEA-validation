import tkinter
from tkinter import *
import tkinter as tk
import customtkinter
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pylab import show, imshow, gray
import os
import pandas as pd
from matplotlib import cm
import math


customtkinter.set_appearance_mode("dark")

degx = 0
degy = 0


# Function to generate PDF

def generate(image1, image2, image3, image4, paragraph_text, output_path, line1, line2, scr, ecr, coefficients_count, rmsexp_values, cbmin, cbmax):
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))

    im1 = axs[0, 0].imshow(image1.get_array(),vmin = cbmin, vmax = cbmax, cmap='viridis')
    axs[0, 0].set_title("Fitted_Exp")
    axs[0, 0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axs[0, 0])
    cbar1.set_label('Colorbar 1 Label')

    im2 = axs[0, 1].imshow(image2.get_array(),vmin = cbmin, vmax = cbmax, cmap='viridis')
    axs[0, 1].set_title("fitted simulation")
    axs[0, 1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axs[0, 1])
    cbar2.set_label('Colorbar 2 Label')

    axs[1, 0].imshow(image3.get_array(), cmap='jet')
    axs[1, 0].set_title("normalized_exp")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(image4.get_array(), cmap='jet')
    axs[1, 1].set_title("normalized_simulation")
    axs[1, 1].axis('off')

    # Plot SCR against ECR with line1 and line2
    axs[2, 0].scatter(ecr, scr, label='SCR vs ECR', color='blue') 
    axs[2, 0].plot(ecr, line1)
    axs[2, 0].plot(ecr, line2)
    axs[2, 0].set_xlabel('Experimental Coefficients')
    axs[2, 0].set_ylabel('Simulation Coefficients')
    axs[2, 0].legend()

    # Plot RMSEXP against coefficients count
    axs[2, 1].plot(coefficients_count, rmsexp_values, 'm')
    axs[2, 1].set_xlabel('Coefficients Count')
    axs[2, 1].set_ylabel('RMSEXP')

    # Leave the fourth row subplot empty for the paragraph
    axs[3, 0].axis('off')
    axs[3, 1].axis('off')

    # Add the paragraph text
    axs[3, 0].text(0.5, 0.5, paragraph_text, ha='center', va='center', wrap=True)

    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.savefig(output_path, format="jpg")
    plt.show()


def calculate_errors(original_image, fitted_image):
    # Calculate Euclidean distance
    original_image = original_image.astype(np.float32)

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    euclidean_distance = np.sqrt(np.sum((original_image - fitted_image)**2))

    # Calculate RMS error
    rms = np.sqrt(np.mean((original_image - fitted_image)**2))

    return euclidean_distance, rms


def chebfit2d(x0, x1, y, deg, rcond=None, full=False, w=None):
    # Generate the 2D Chebyshev Vandermonde matrix
    a = np.polynomial.chebyshev.chebvander2d(x0, x1, deg)

    # Solve the linear least squares problem using numpy's lstsq function
    x, residuals, rank, s = np.linalg.lstsq(a, y, rcond=rcond)

    # If 'full' is True, return additional information about the solution
    if full:
        return x.reshape(deg[0]+1, deg[1]+1), [residuals, rank, s, rcond]

    # Return the fitted coefficients reshaped into a 2D array
    return x.reshape(deg[0]+1, deg[1]+1)


def fit_chebyshev_to_image(image_array, degree=(degx, degy)):
    # Get the dimensions of the image
    normalized_displacement_image_32f = image_array.astype(np.float32)
    grayscale_image = cv2.cvtColor(normalized_displacement_image_32f, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape[:2]

    # Generate pixel coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)

    # Flatten the coordinates and pixel values
    x_flat = x.flatten()
    y_flat = y.flatten()
    image_flat = grayscale_image.flatten()

    # Use the chebfit2d function to fit the polynomial
    coefficients = chebfit2d(x_flat, y_flat, image_flat, degree)

    # Evaluate the fitted polynomial on a grid
    fit_result = np.polynomial.chebyshev.chebval2d(x, y, coefficients)
    return fit_result, coefficients


def normalize_image_with_colorbar(original_image, color_bar, high, low):
    color_bar_image = np.array(color_bar)
    original_image = np.array(original_image)

    # Load original image
    if original_image is None:
        raise FileNotFoundError(f"Failed to load original image '{original_image}'")

    # Load color bar image
    if color_bar_image is None:
        raise FileNotFoundError(f"Failed to load color bar image '{color_bar_image}'")

    # Determine if the color bar is horizontal or vertical
    if color_bar_image.shape[0] > color_bar_image.shape[1]:
        is_horizontal = False
        num_rows = color_bar_image.shape[0]
        num_cols = 1
    else:
        is_horizontal = True
        num_rows = 1
        num_cols = color_bar_image.shape[1]

    # Define the range of values corresponding to the color bar
    color_bar_values = np.linspace(high, low, num_cols if is_horizontal else num_rows)

    # Create a dictionary to map RGB values to corresponding values
    color_value_dict = {}

    # Iterate over each row or column in the color bar
    if is_horizontal:
        for col_index in range(num_cols):
            rgb_values = color_bar_image[0, col_index]
            color_value_dict[tuple(rgb_values)] = color_bar_values[col_index]
    else:
        for row_index in range(num_rows):
            rgb_values = color_bar_image[row_index, 0]
            color_value_dict[tuple(rgb_values)] = color_bar_values[row_index]

    # Create a KDTree for nearest neighbor search
    tree = KDTree(list(color_value_dict.keys()))

    # Smooth original image
    smoothed_image = original_image

    # Initialize an empty array for the displacement values
    displacement_image = np.zeros_like(smoothed_image, dtype=float)

    # Map RGB values to displacement values
    for row in range(smoothed_image.shape[0]):
        for col in range(smoothed_image.shape[1]):
            pixel_rgb = tuple(smoothed_image[row, col])
            _, result = tree.query(pixel_rgb)  # Find the nearest color in the color bar
            nearest_color = tree.data[result]
            displacement_image[row, col] = color_value_dict[tuple(nearest_color)]

    normalized_displacement_image = displacement_image.astype(np.float32)
    return normalized_displacement_image


def extract_color_bar_and_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hue = 0
    low_saturation = 80
    low_value = 0
    high_hue = 255
    high_saturation = 255
    high_value = 255
    # Define the rae of colors that correspond to the color bar
    lower_color = np.array([low_hue, low_saturation, low_value], dtype=np.uint8)
    upper_color = np.array([high_hue, high_saturation, high_value], dtype=np.uint8)
    # Threshold the image to get a binary mask
    color_bar_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(color_bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
    # Find the contour with the smallest area and largest area (assuming it's the color bar)
    if contours:
        color_bar_contour = min(contours, key=cv2.contourArea)
        # Get the bounding box of the color bar
        x, y, w, h = cv2.boundingRect(color_bar_contour)
        # Crop the color bar region from the original image
        color_bar_region = image[y:y + h, x:x + w]
        # Convert color bar region to PIL Image
        color_bar_image = Image.fromarray(cv2.cvtColor(color_bar_region, cv2.COLOR_BGR2RGB))
        # Find the contour with the largest area for the main image
        image_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of the main image
        x, y, w, h = cv2.boundingRect(image_contour)
        # Crop the main image region from the original image
        image_region = image[y:y + h, x:x + w]
        # Convert main image region to PIL Image
        result_image = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
        return color_bar_image, result_image


def extract_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hue = 0
    low_saturation = 80
    low_value = 0
    high_hue = 255
    high_saturation = 255
    high_value = 255
    # Define the rae of colors that correspond to the color bar
    lower_color = np.array([low_hue, low_saturation, low_value], dtype=np.uint8)
    upper_color = np.array([high_hue, high_saturation, high_value], dtype=np.uint8)

    # Threshold the image to get a binary mask
    color_bar_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(color_bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
    # Find the contour with the largest area (assuming it's the color bar)
    if contours:
        # Find the contour with the largest area for the main image
        image_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the main image
        x, y, w, h = cv2.boundingRect(image_contour)

        # Crop the main image region from the original image
        image_region = image[y:y + h, x:x + w]

        # Convert main image region to PIL Image
        result_image = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
        return result_image

    else:
        return "Image Not found"


class MyFrame(customtkinter.CTkFrame):
    def submit_and_destroy(self):
        self.master.sim_high_value, self.master.sim_low_value, self.master.exp_high_value, self.master.exp_low_value, self.master.exp_error, self.master.checked_box, self.master.coefficients = self.get_high_low_values()
        self.master.destroy()

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Add widgets onto the frame...
        # TextBox
        self.textbox2 = customtkinter.CTkLabel(self)
        self.textbox2.grid(row=1, column=0, padx=22)
        self.textbox2.configure(text="Simulation Image")
        # Input file
        self.Simulation_image = customtkinter.CTkEntry(self, width=250, placeholder_text='Input Simulation Image')
        self.Simulation_image.Simulation_path = ""
        self.Simulation_image.grid(row=1, column=1, padx=22, pady=3)
        # Input file browse
        self.Simulation_image_button = customtkinter.CTkButton(self, text="Browse", width = 100, command=self.Input_Simulation_Result)
        self.Simulation_image_button.grid(row=1, column=4, padx=22, pady=3)

        # Add high and low input values for Simulation Image
        self.Simulation_high_value = customtkinter.CTkEntry(self, width=50, placeholder_text='High')
        self.Simulation_high_value.grid(row=1, column=2, padx=5, pady=3)
        self.Simulation_low_value = customtkinter.CTkEntry(self, width=50, placeholder_text='Low')
        self.Simulation_low_value.grid(row=1, column=3, padx=5, pady=3)

        # TextBox
        self.Textbox1 = customtkinter.CTkLabel(self)
        self.Textbox1.grid(row=2, column=0, padx=22)
        self.Textbox1.configure(text="Experimental Image")
        # Input file
        self.Experimental_image = customtkinter.CTkEntry(self, width=250, placeholder_text='Input Experimental Image')
        self.Experimental_image.Experimental_path = ""
        self.Experimental_image.grid(row=2, column=1, padx=22, pady=3)
        # Input file browse
        self.Experimental_image_button = customtkinter.CTkButton(self, text="Browse", width = 100, command=self.Input_Experimental_Result)
        self.Experimental_image_button.grid(row=2, column=4, padx=22, pady=3)

        # Add high and low input values for Experimental Image
        self.Experimental_high_value = customtkinter.CTkEntry(self, width=50, placeholder_text='High')
        self.Experimental_high_value.grid(row=2, column=2, padx=5, pady=3)
        self.Experimental_low_value = customtkinter.CTkEntry(self, width=50, placeholder_text='Low')
        self.Experimental_low_value.grid(row=2, column=3, padx=5, pady=3)

        # Add input for Experimental Error
        self.exp_error = customtkinter.CTkEntry(self, width=250, placeholder_text='Experimental Error')
        self.exp_error.grid(row=4, column=1, padx=5, pady=3)

        self.Textbox5 = customtkinter.CTkLabel(self)
        self.Textbox5.grid(row=4, column=0, padx=22)
        self.Textbox5.configure(text="minimum experiment uncertainty?")

        self.Textbox5 = customtkinter.CTkLabel(self)
        self.Textbox5.grid(row=5, column=0, padx=22)
        self.Textbox5.configure(text="Colormap in picture?")

        self.cm = customtkinter.CTkCheckBox(self)
        self.cm.grid(row=5, column=1, padx=22)
        self.cm.configure(text="")

        self.Textbox6 = customtkinter.CTkLabel(self)
        self.Textbox6.grid(row=6, column=0, padx=22)
        self.Textbox6.configure(text="Save output coefficients?")

        self.cs = customtkinter.CTkCheckBox(self)
        self.cs.grid(row=6, column=1, padx=22)
        self.cs.configure(text="")

        self.submit_button = customtkinter.CTkButton(self)
        self.submit_button.grid(row=7, column=1, padx=22, pady=3, sticky="nsew")
        self.submit_button.configure(text="Submit", command=self.submit_and_destroy)

    def get_high_low_values(self):
        sim_high = self.Simulation_high_value.get()
        sim_low = self.Simulation_low_value.get()
        exp_high = self.Experimental_high_value.get()
        exp_low = self.Experimental_low_value.get()
        exp_error = self.exp_error.get()
        checked_box = self.cm.get()
        coefficients = self.cs.get()
        return sim_high, sim_low, exp_high, exp_low, exp_error, checked_box, coefficients

    def Input_Simulation_Result(self):
        Simulation_path = filedialog.askopenfilename()
        self.Simulation_image.delete(0, END)
        self.Simulation_image.insert(0, Simulation_path)
        self.Simulation_image.Simulation_path = Simulation_path

    def Input_Experimental_Result(self):
        Experimental_path = filedialog.askopenfilename()
        self.Experimental_image.delete(0, END)
        self.Experimental_image.insert(0, Experimental_path)
        self.Experimental_image.Experimental_path = Experimental_path


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("830x350")
        self.grid_rowconfigure(0, weight=1)  # configure grid system
        self.grid_columnconfigure(0, weight=1)
        self.title("Validation Methodology")
        self.my_frame = MyFrame(master=self)
        self.my_frame.grid(row=0, column=0, padx=22, pady=22, sticky="nsew")


app = App()
app.mainloop()
sim_high_value, sim_low_value, exp_high_value, exp_low_value, exp_error, cs = float(app.sim_high_value), float(app.sim_low_value), float(app.exp_high_value), float(app.exp_low_value), float(app.exp_error), float(app.coefficients)
simimage = app.my_frame.Simulation_image.Simulation_path
expimage = app.my_frame.Experimental_image.Experimental_path
cmb = app.checked_box
imglists = [simimage, expimage]
simimage = cv2.imread(simimage)
expimage = cv2.imread(expimage)

if cmb == 1:

    simcolorbar, simulation_image = extract_color_bar_and_image(simimage)
    colorbar, experimental_image = extract_color_bar_and_image(expimage)

    normalized_sim_image = normalize_image_with_colorbar(simulation_image, simcolorbar, sim_high_value, sim_low_value)
    normalized_exp_image = normalize_image_with_colorbar(experimental_image, colorbar, exp_high_value, exp_low_value)

elif cmb == 0:
    raise Exception("Colorbar needs to be in image")

normalized_exp_image1 = normalized_exp_image.astype(np.float32)
normalized_exp_image1 = cv2.cvtColor(normalized_exp_image1, cv2.COLOR_BGR2GRAY)

normalized_sim_image1 = normalized_sim_image.astype(np.float32)
normalized_sim_image1 = cv2.cvtColor(normalized_sim_image1, cv2.COLOR_BGR2GRAY)

sim_height, sim_width, _ = simimage.shape
exp_height, exp_width, _ = expimage.shape

if abs(sim_height * sim_width - exp_height * exp_width) > 0.2 * (exp_height * exp_width) or (sim_height * sim_width < 0.8 * (exp_height * exp_width)):
    raise Exception("Simulation image size is not within 20% of the size of experimental image.")

rmsexp = 10
rmsexp_values = []  # To store rmsexp values
eucexp_values = []  # To store eucexp values
coefficients_count = []  # To store number of coefficients

max_coefficients = 1000
while rmsexp > exp_error:
    # Update the degree parameters
    degx += 1
    degy += 1

    # Fit Chebyshev to the images with updated degrees
    fitted_image_sim, simcoefficients = fit_chebyshev_to_image(normalized_sim_image, degree=(degx, degy))
    fitted_image_exp, expcoefficients = fit_chebyshev_to_image(normalized_exp_image, degree=(degx, degy))
    # Calculate errors for the experimental image
    eucexp, rmsexp = calculate_errors(normalized_exp_image, fitted_image_exp)
    coefficients_count.append(len(expcoefficients.ravel()))
    rmsexp_values.append(rmsexp)
    eucexp_values.append(eucexp)

    if len(expcoefficients.ravel()) > max_coefficients:
        break

scr = simcoefficients.ravel()
ecr = expcoefficients.ravel()
rms = np.sqrt(np.mean((scr - ecr)**2))

output_df = pd.DataFrame()

if cs == 1:
    output_df['simulation'] = scr
    output_df['experiment'] = ecr
    output_df.to_csv('Coefficents.csv')

# Define the length of the data (n) and the arrays scr and ecr
n = len(scr)

# Calculate TC1
TC2 = math.sqrt(sum((scr[k] - ecr[k])**2 for k in range(n))) / (math.sqrt(sum(ecr[k]**2 for k in range(n))))

n = len(ecr)


# Function to calculate TC1
def calculate_tc2(ecr, scr, n):
    return math.sqrt(sum((scr[k] - ecr[k]) ** 2 for k in range(n))) / math.sqrt(sum(ecr[k] ** 2 for k in range(n)))


# Calculate TC1 for the original ecr
TC1_original = calculate_tc2(ecr, scr, n)

# Calculate TC1 for ecr[0] + uexp
ecr_plus = ecr.copy()
ecr_plus[0] += exp_error
TC1_plus_uexp = calculate_tc2(ecr_plus, scr, n)

# Calculate TC1 for ecr[0] - uexp
ecr_minus = ecr.copy()
ecr_minus[0] -= exp_error
TC1_minus_uexp = calculate_tc2(ecr_minus, scr, n)

# Data for plotting
tc_values = [(1 - TC1_minus_uexp)*100, (1 - TC1_original)*100, (1 - TC1_plus_uexp) * 100]
labels = ['TC1 - uexp', 'TC1 Original', 'TC1 + uexp']


ek = abs((scr - ecr)/ecr.max())
wk = ek / np.sum(ek) * 100
sum_wk = 0
uexp = math.sqrt(exp_error**2 + rmsexp**2)
eunc = 2*uexp/ecr.max()*100

# Iterate over each element in ek and wk simultaneously
for ek_val, wk_val in zip(ek*100, wk):
    # Check if ek is less than exp_error
    if ek_val < eunc:
        # Add the corresponding wk to sum_wk
        sum_wk += wk_val

fis = plt.imshow(fitted_image_sim, cmap='jet')
fie = plt.imshow(fitted_image_exp, cmap='jet')
nis = plt.imshow(simulation_image, cmap='jet')
nie = plt.imshow(experimental_image, cmap='jet')

line1 = ecr + 2*uexp
line2 = ecr - 2*uexp
# Generate PDF
if np.any((scr > line1) | (scr < line2)):
    text = "CEN VM: Invalid"
else:
    text = "CEN VM: Valid"

output_path = "result.jpg"
info_text = "Theil's inequality coefficient is {:.4f}\nRelative Error Validation Metric is {:.1f} \nNumber of coefficients {:.0f}\n{}\n Proposed VM is between {:.2f}% and {:.2f}%".format(TC2, round(sum_wk, 1), len(scr), text, min(tc_values), max(tc_values))
sum_wk = 0

sim_min_value = np.min(fitted_image_sim)
sim_max_value = np.max(fitted_image_sim)

# Calculate the minimum and maximum values for normalized_exp_image
exp_min_value = np.min(fitted_image_exp)
exp_max_value = np.max(fitted_image_exp)

# Now, assign these values to cblists
cblists = [sim_max_value, sim_min_value, exp_max_value, exp_min_value]
generate(fie, fis, nie, nis, info_text, output_path, line1, line2, scr, ecr, coefficients_count, rmsexp_values, min(cblists), max(cblists)) 
