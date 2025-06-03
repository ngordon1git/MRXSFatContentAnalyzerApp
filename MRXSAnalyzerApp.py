import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import json
import csv
from pathlib import Path
from FatAnalyzerParametersSelectionUI import FatAnalyzerParametersSelectionUI
from HistomicksTK_class import LiverFatAnalyzer

class MRXSAnalyzerApp ():
    def __init__(self, master):
        self.master = master
        self.master.title("MRXS Analyzer")

        # State
        self.folder_path = tk.StringVar()
        self.params_path = tk.StringVar()
        self.mrxs_files = []
        # self.bounds_selected = set()
        self.params = {}
        self.selected_file = tk.StringVar()
        self.reanalyze_all = tk.BooleanVar()
        self.create_histograms = tk.BooleanVar(value=True)

        self.results_df = None
        # Analyzer placeholder
        self.fat_analyzer = None

        self.setup_ui()
        # TODO this is for debugging only
        # self.folder_path.set('/Users/natangordon/PycharmProjects/OpenSlide_Hepatic_Fat')
        # self.load_existing_data()
        # self.find_mrxs_files()
        # self.update_file_list()

    def setup_ui(self):
        frame = ttk.Frame(self.master, padding=10)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(1, weight=1)

        # Folder & Params File Selection
        ttk.Button(frame, text="Choose Folder", command=self.choose_folder).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.folder_path).grid(row=0, column=1, sticky="w")

        ttk.Button(frame, text="Choose Params File", command=self.choose_params_file).grid(row=1, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.params_path).grid(row=1, column=1, sticky="w")

        # MRXS file list with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.file_listbox = tk.Listbox(list_frame, width=60, height=10, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.file_listbox.yview)

        self.file_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # Buttons
        ttk.Button(frame, text="Select Bounds", command=self.select_bounds).grid(row=3, column=0, pady=5, sticky="w")
        ttk.Button(frame, text="Select Params", command=self.select_params).grid(row=3, column=1, pady=5, sticky="w")

        # Checkboxes
        ttk.Checkbutton(frame, text="Re-analyze all", variable=self.reanalyze_all).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(frame, text="Create Histograms", variable=self.create_histograms).grid(row=4, column=1, sticky="w")

        # Run
        self.run_btn = ttk.Button(frame, text="Run", command=self.run_analysis, state=tk.DISABLED)
        self.run_btn.grid(row=5, column=0, columnspan=2, pady=10)

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.folder_path.set(folder)
        self.load_existing_data()
        self.find_mrxs_files()
        self.update_file_list()

    def choose_params_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not path:
            return

        self.params_path.set(path)
        with open(path, 'r') as f:
            self.params = json.load(f)
        self.save_params()

    def load_existing_data(self):
        self.results_dir = Path(self.folder_path.get()) / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.results_path = self.results_dir / "results.csv"

        if self.results_path.exists():
            self.results_df = pd.read_csv(self.results_path)
        else:
            self.results_df = pd.DataFrame(columns=[
                "file_path", "x0", "y0", "width", "height", "total_tissue", "total_fat", 'fat_content','fat_content_std'
            ])

        params_file_path = self.results_dir / "params.json"
        if (params_file_path).exists():
            with open(params_file_path, 'r') as f:
                self.params = json.load(f)
                self.params_path.set(params_file_path)

    def save_params(self):
        results_dir = Path(self.folder_path.get()) / "results"
        params_file_path = results_dir / "params.json"
        with open(params_file_path, 'w') as f:
            json.dump(self.params, f, indent=2)
            self.params_path.set(params_file_path)

    def save_bounds(self, file_path, bounds):
        """
        Save or update bounds for a file.
        bounds = {"x0": int, "y0": int, "width": int, "height": int}
        """
        # Remove existing row for file if exists
        self.results_df = self.results_df[self.results_df["file_path"] != file_path]
        x0, y0, width, height = bounds
        new_row = { # note: new bounds means new results. Therefore, deleting old results.
            "file_path": file_path,
            "x0": x0,
            "y0": y0,
            "width": width,
            "height": height,
            'total_tissue': None,
            'total_fat': None,
            "fat_content": None,
            "fat_content_std": None
        }

        self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)
        self.results_df.to_csv(self.results_path, index=False)

    def save_results(self, file_path, results):
        """
        Save or update results for a given file.
        results = {
            "total_tissue": int,
            "total_fat": int,
            "fat_content": float,
            "fat_content_std": float
        }
        """
        if self.results_df.empty or "file_path" not in self.results_df.columns:
            raise ValueError("Results dataframe not initialized or malformed.")

        # Check if file exists in the dataframe
        mask = self.results_df["file_path"] == file_path
        if mask.any():
            # Update existing row in-place
            for key in ["total_tissue", "total_fat", "fat_content", "fat_content_std"]:
                self.results_df.loc[mask, key] = float(results.get(key))

        self.results_df.to_csv(self.results_path, index=False)

    def get_bounds(self, file_path):
        row = self.results_df[self.results_df["file_path"] == file_path]
        if row.empty:
            return None
        if row[["x0", "y0", "width", "height"]].isnull().any(axis=1).values[0]:
            return None
        return {
            "x0": int(row["x0"].values[0]),
            "y0": int(row["y0"].values[0]),
            "width": int(row["width"].values[0]),
            "height": int(row["height"].values[0])
        }

    def has_results(self, file_path):
        if not os.path.exists(file_path): return False
        row = self.results_df[self.results_df["file_path"] == file_path]
        if row.empty:
            return False
        return not row[["fat_content", "fat_content_std"]].isnull().any(axis=1).values[0]

    def find_mrxs_files(self):
        self.mrxs_files.clear()
        for root, _, files in os.walk(self.folder_path.get()):
            for file in files:
                if file.lower().endswith(".mrxs"):
                    full_path = os.path.join(root, file)
                    self.mrxs_files.append(full_path)

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for path in self.mrxs_files:
            name = os.path.basename(path)
            idx = self.file_listbox.size()
            self.file_listbox.insert(idx, name)
            if self.has_results(path):
                self.file_listbox.itemconfig(idx, foreground="green")  # Optional: mark with results only
            elif self.get_bounds(path):
                self.file_listbox.itemconfig(idx, foreground="blue")
        # This is also a good place to call any other update, such as:
        run_button_state = tk.DISABLED if (self.params == {} or self.params is None) else tk.NORMAL
        self.run_btn.config(state = run_button_state)

    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.selected_file.set(self.mrxs_files[idx])

    def select_bounds(self, file_path = None):
        if file_path is None:
            if not self.selected_file.get():
                return
            file_path = self.selected_file.get()

        print(f"[SELECT BOUNDS] for {file_path}")
        # Bounds selection
        liver_fat_analyzer = LiverFatAnalyzer(file_path)
        selected_bounds = liver_fat_analyzer.display_thumbnail_for_bounds_selection()
        print('Bounds seleceted: ', selected_bounds)

        # self.bounds_selected.add(file_path)
        self.save_bounds(file_path=file_path, bounds=selected_bounds)
        self.update_file_list()

    def select_params(self):
        print("[SELECT PARAMS] Called")

        if not self.selected_file.get():
            return
        file_path = self.selected_file.get()
        fat_analyzer = FatAnalyzerParametersSelectionUI(self.master, file_path,
                                                        callback=self.update_params)
        fat_analyzer.wait_window()
        self.save_params()

        # ----  While selecting new params, we've also selected new bounds!
        new_selected_bounds = fat_analyzer.liver_fat_analyzer.slide_bounds
        self.save_bounds(file_path=file_path, bounds=new_selected_bounds)
        self.update_file_list()

    def run_analysis(self):
        # Disable the run button
        self.run_btn.config(state=tk.DISABLED,text="Running...")

        if self.params is None: return
        analyze_all = self.reanalyze_all.get()
        create_histograms = self.create_histograms.get()
        print("Running analysis...")
        print("Folder:", self.folder_path.get())
        print("Params:", self.params)
        print("Re-analyze all:", analyze_all)
        print("Create Histograms:", create_histograms)

        for file_path in self.mrxs_files: # For each file in the list
            if not self.get_bounds(file_path) or analyze_all:
                self.select_bounds(file_path=file_path)

        # Now all bounds are selected, and we should be able to continue with analysis
        # Note we do have values in self.params
        if not self.has_results(file_path) or analyze_all:
            print("Analyzing ", file_path, '...')
            # Create the path for results of this specific file
            file_name = os.path.basename(file_path)
            cur_file_results_path = Path(self.results_dir) / file_name
            cur_file_results_path.mkdir(exist_ok=True) # Create path if it doesn't exist
            # Create analyzer, set its bounds and params
            fat_analyzer = LiverFatAnalyzer(slide_path=file_path,slide_bounds = self.get_bounds(file_path),
                                            output_dir=cur_file_results_path)
            fat_analyzer.set_params_from_dic(self.params)
            fat_analyzer.process_slide()
            fat_analyzer.export_csv(cur_file_results_path / 'results.csv') # Export results for specific file
            self.save_results(file_path=file_path, results=fat_analyzer.summarize()) # Update summary file
            if create_histograms:
                fat_analyzer.export_fat_content_histogram(output_path=cur_file_results_path / 'fat_content_histogram.png', bins =20)

        self.update_file_list()
        # Re-enable window and show message
        self.run_btn.config(state=tk.NORMAL,text="Run")
        tk.messagebox.showinfo("Done", "Analysis done!")


    def update_params(self, p):
        self.params = p
        print('Params updated: ', p)

    def _print(self, s):
        print(s)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MRXSAnalyzerApp(root)
    root.mainloop()
