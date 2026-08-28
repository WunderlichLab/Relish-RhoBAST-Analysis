# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:26:18 2024

Name: Emma Rits
Date: Monday, August 12, 2024
Description: Manual trace behavior sorting interface

"""

#%% import packages and functions

# packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk

datetime_str = time.strftime("%m%d%y_%H:%M")

# functions
from step9_Relish_Trace_PreProcessing_python import save_data, import_data, flatten_trace_df


#%% functions to build GUI

trace_behaviors = ["Nonresponsive", "Delayed", "Gradual", "Immediate", "Immediate with decay"]

class CellTraceGUI:
    def __init__(self, root, df, categories, file_path, stim_time = 30, AMP = None):
        self.root = root
        self.root.title("Cell Trace Categorization")

        self.df = df
        self.file_path  = file_path
        self.stim_time  = stim_time
        self.AMP        = AMP
        self.timepoints = list(df.columns)

        # generate list of cells to test
        self.num_cells = len(self.df)
        self.current_index = 0
        self.cell_indices = np.random.choice(df.index, self.num_cells, replace = False)
        
        # initialize dictionary to track counts
        self.categories = categories
        self.cell_categories = {category: [] for category in categories}

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.plot_trace()

        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(root, textvariable=self.category_var, values=self.categories)
        self.category_dropdown.pack(pady=10)

        self.next_button = tk.Button(root, text="Next Cell", command=self.next_trace)
        self.next_button.pack(pady=5)
        
        self.save_button = tk.Button(root, text="Save Category", command=self.save_category)
        self.save_button.pack(pady=5)
        
        self.save_data_button = tk.Button(root, text="Save Data", command=self.save_data)
        self.save_data_button.pack(pady=5)
        
        self.counter_label = tk.Label(root, text=self.get_counter_text(), anchor='w')
        self.counter_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    
    
    def plot_trace(self):
        self.ax.clear()
        cell_index = self.cell_indices[self.current_index]
        
        # ensure cell_index is a string and valid
        if not isinstance(cell_index, str):
            raise ValueError("Cell indices must be formatted as strings.")

        if cell_index in self.df.index:
            cell_data = self.df.loc[cell_index].values
            self.ax.plot(self.timepoints, cell_data)
            self.ax.set_title(f"{self.cell_indices[self.current_index]}", fontsize = 16)
            self.ax.set_xlabel("Time (min)", fontsize = 14)
            self.ax.set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
            self.ax.tick_params(axis = "both", labelsize = 12)
            
            # show major gridlines
            self.ax.grid(which = "major", color = "lightgrey", linestyle = "dashed", linewidth = 0.5)
            
            # plot dashed lines at t = 30 and y = 1, 1.2
            plt.axvline(x = self.stim_time, color = "black", linewidth = 0.5)
            plt.axvline(x = self.stim_time + 120, color = "black", linewidth = 0.5)
            plt.axhline(y = 1, color = "black", linewidth = 0.5)
            # plt.axhline(y = 1.2, color = "black", linewidth = 0.5)
            
            # set y-axis limits and ticks
            self.ax.set_ylim(0.6, 2.0)
            self.ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
            
            # display key values
            init_val = cell_data[0]
            hun_val  = cell_data[self.stim_time + 120]
            max_val  = cell_data.max()
            fin_val  = cell_data[-1]
            self.ax.text(
                860.0,                                                          # x position (relative to x-axis range)
                1.9,                                                            # y position (relative to y-axis range)
                f'$t_{{0}}$: {init_val:.3f}\n$t_{{{self.stim_time + 120}}}$: {hun_val:.3f}\n $t_{{{self.timepoints[-1]}}}$: {fin_val:.3f}\n\nMax: {max_val:.3f}',
                horizontalalignment = 'right',
                verticalalignment = 'top',
                fontsize = 14,
                color = 'black',
                bbox = dict(facecolor='white', edgecolor='none', pad=2)
            )
            
            self.canvas.draw()
        else:
            raise ValueError(f"{cell_index} not found.")
    
    
    def save_category(self):
        category = self.category_var.get()
        
        if category:
            cell_index = self.cell_indices[self.current_index]
            self.cell_categories[category].append(cell_index)
            self.category_var.set("")
            self.update_counter()
            
            # Check if any category has reached 20 traces
            if all(len(cells) >= 20 for cells in self.cell_categories.values()):
                print("20 traces classified into each category. Saving data and exiting.")
                self.save_data()
                self.root.quit()                                                # quit the Tkinter main loop
            
            self.next_trace()                                                   # automatically move to the next trace
        
        else:
            print("Please select a category.")
    
    
    def next_trace(self):
        if self.current_index < len(self.cell_indices) - 1:
            self.current_index += 1
            self.plot_trace()
        else:
            print("No more cells to display.")
            self.save_data()
    
    
    def save_data(self):
        # save df to CSV
        df_categories = self.get_categories()
        csv_file_path = 'cell_categories.csv'
        df_categories.to_csv(csv_file_path, index=False)
        print(f"Categories saved to '{csv_file_path}'.")

        # save df as a pickle object
        object_name = "trace_sorting_cell_categories" if self.AMP == None else f"trace_sorting_cell_categories_{self.AMP}"
        save_data(file_path = self.file_path, object_data = df_categories, object_name = object_name)
        full_file_path = f"{self.file_path}\\{object_name}.pkl"
        print(f"Data saved as .pkl file at: {full_file_path}.")


    def get_categories(self):
        self.df_categories = pd.DataFrame(list(self.cell_categories.items()), columns=["Category", "Cells"])
        # print(df_categories)
        return self.df_categories
    
    
    def update_counter(self):
        self.counter_label.config(text=self.get_counter_text())
    
    
    def get_counter_text(self):
        counts = {cat: len(self.cell_categories[cat]) for cat in self.categories}
        total = sum(counts.values())
        counts["Total"] = total
        counter_text = "Counts: " + ", ".join(f"{cat}: {count}" for cat, count in counts.items())
        return counter_text


#%% # === MAIN LOOP ===

# file paths
all_data         = "/path/to/your/data/2025-01-01_datasetName/"
file_path_step9  = all_data + "Processed Traces"
file_path_step10 = all_data + "SVM Classifier"
all_traces_df    = import_data(file_path_step9, "all_traces_df")

# run GUI
root = tk.Tk()
app = CellTraceGUI(root, all_traces_df, trace_behaviors, file_path_step10)
root.mainloop()
behavior_categories_df = app.df_categories
save_data(file_path_step10, behavior_categories_df, "behavior_categories_df")
