import sys
from FingerprintImageEnhancer import FingerprintImageEnhancer
import cv2
import os
from tkinter import *
from tkinter import ttk
import numpy as np
import natsort

from tkinter import filedialog
from tkinter import messagebox 
from tkinter.ttk import Progressbar
import tkinter.font as font

src_path = ''
dest_path = ''

def GUI():
	main_window = Tk()

	main_window.title("Fingerprint_Enhancer")
	main_window.geometry('600x300')
	main_window.resizable(False, False)
	main_window.configure(background='white')

	btn_font = font.Font(family='Helvetica', size=12, weight='bold')
	prg_style = ttk.Style()
	prg_style.configure("bar.Horizontal.TProgressbar", troughcolor='#FFFCFF', background='#0FD64F')

##########################################################################################################

	source_path_label = Label(main_window, text="Enter source folder path: ", bg='white')
	source_path_label.grid(column=0, row=0, padx = 15, pady= 15)

	source_path_value = Label(main_window, text = 'None', width=30, bg='white', borderwidth=1, relief='solid')
	source_path_value.grid(column=1, row=0, padx = 15, pady= 15)

	source_browse_btn = Button(main_window, bg='#09C6F9', activebackground='#045DE9', fg='white', activeforeground= 'white', borderwidth = 0, text='Browse', font= btn_font, command=lambda:browse_folder(True, main_window, source_path_value))
	source_browse_btn.grid(column=2, row=0, padx = 15, pady= 15)

###########################################################################################################

	dest_path_label = Label(main_window, text="Enter destin folder path: ", bg='white')
	dest_path_label.grid(column=0, row=1, padx = 15, pady= 15)

	dest_path_value = Label(main_window, text= 'None', width=30, bg='white', borderwidth=1, relief='solid')
	dest_path_value.grid(column=1, row=1, padx = 15, pady= 15)

	dest_browse_btn = Button(main_window, bg='#09C6F9', activebackground='#045DE9', fg='white', activeforeground= 'white', borderwidth = 0, text='Browse', font=btn_font,command=lambda:browse_folder(False, main_window, dest_path_value))
	dest_browse_btn.grid(column=2, row=1, padx = 15, pady= 15)


###########################################################################################################

	run_btn = Button(main_window, bg='#0FD64F', activebackground='#00B712', fg='white', activeforeground= 'white', borderwidth = 0, text = 'Run', font=btn_font, command = lambda:check_paths(progress_bar, prg_percent, main_window))
	run_btn.grid(column=1, row=2, padx = 15, pady= 20)

	close_btn = Button(main_window, bg='#F9484A', activebackground='#F9484A', fg='white', activeforeground= 'white', borderwidth = 0, text = 'Close', font=btn_font, command = lambda:main_window.destroy())
	close_btn.grid(column=2, row=2, padx = 15, pady= 20)

############################################################################################################
	prg_percent = Label(main_window, text="0%", bg='white')
	prg_percent.place(x=300, y=220)

	progress_bar = Progressbar(main_window, orient=HORIZONTAL, style= 'bar.Horizontal.TProgressbar', length = 100, mode='determinate')
	progress_bar.place(x=15, y=250, height=20, width=575)

###########################################################################################################
	
	main_window.mainloop()

def browse_folder(isSource, main_window, entry):
	global src_path, dest_path
	if isSource:
		src_path = filedialog.askdirectory(parent= main_window, title= 'Choose source folder')
		entry.config(text = src_path)
	else:
		dest_path = filedialog.askdirectory(parent= main_window, title= 'Choose destination folder')
		entry.config(text = dest_path)

def check_paths(prg_bar, prg_percent, main_window):
	global src_path, dest_path

	if src_path or dest_path:
		if os.listdir(src_path):
			
			run_algorithm(prg_bar, prg_percent, main_window)

		else:
			messagebox.showerror("Error", "Source Folder is Empty")
	else:
		messagebox.showerror("Error", "Choose both folders")

def run_algorithm(prg_bar, prg_percent, main_window):
	global src_path, dest_path

	src_path = src_path + '/'
	dest_path = dest_path + '/'

	image_enhancer = FingerprintImageEnhancer() 

	for prg,image in enumerate(natsort.natsorted(os.listdir(src_path))):
		if '.jpg' in image or '.PNG' in image:
			img = cv2.imread(os.path.join(src_path + image))

			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			out_img = image_enhancer.enhance(img)

			image_enhancer.save_enhanced_image(dest_path + image.split('.')[0] + '.bmp')
			
			per = ((prg + 1) / len(os.listdir(src_path))) *100
			prg_bar['value'] = per
			prg_percent.config(text=str(per)+"%")

			main_window.update_idletasks()
		else:
			print("Wrong file")
	messagebox.showinfo("Message", "All photos are enhanced")

if __name__ == '__main__':
	
	GUI()
	
		

