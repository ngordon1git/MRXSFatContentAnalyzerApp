import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from HistomicksTK_class import LiverFatAnalyzer

class FatAnalyzerParametersSelectionUI(tk.Toplevel):
    def __init__(self, master, slide_path, callback = None):
        super().__init__()

        self.callback = callback

        # self.master = master # CHATGPT says remove
        # self.master.title("Fat Analyzer Configurator")
        self.title("Fat Analyzer Configurator")

        # Load image and prepare tiles
        self.liver_fat_analyzer = LiverFatAnalyzer(slide_path)
        self.liver_fat_analyzer.display_thumbnail_for_bounds_selection()
        self.tiles = self.liver_fat_analyzer.get_tiles(only_tiles_to_save=True)

        # Default parameters
        self.params = {
            "show_analysis": tk.BooleanVar(value=True),
            "filter_roundness": tk.BooleanVar(value=True),
            "filter_roundness_threshold": tk.DoubleVar(value=0.75),
            "filter_min_prop_area": tk.DoubleVar(value=1.0),
            "filter_tissue_mask_holes_area": tk.DoubleVar(value=1000.0),
            "filter_tissue_mask_small_object_min_size": tk.DoubleVar(value=200.0),
            "tile_index": tk.IntVar(value=0)
        }

        self.setup_ui()
        self.update_image()

        if self.callback:
            self.grab_set()  # Modal: disables main window
            # self.master.grab_set()  # Modal: disables main window
            # Wait until this window is closed
            self.protocol("WM_DELETE_WINDOW", self.on_cancel)


    def setup_ui(self):
        # main_frame = ttk.Frame(self.master)
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # Left panel: Controls
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side="left", fill="y")

        # Helper function to add label, entry, slider trio
        def add_slider_with_entry(label, var, from_, to_, step=1.0):
            ttk.Label(control_frame, text=label).pack(anchor="w")

            frame = ttk.Frame(control_frame)
            frame.pack(fill="x", pady=5)

            slider = ttk.Scale(frame, from_=from_, to=to_, variable=var,
                               orient="horizontal", command=lambda e: on_slider_change(var, entry, float(e)))
            slider.pack(side="left", fill="x", expand=True)

            entry = ttk.Entry(frame, width=6)
            entry.insert(0, str(var.get()))
            entry.pack(side="right", padx=5)

            def on_entry_change(event):
                try:
                    val = float(entry.get())
                    val = max(from_, min(to_, val))
                    var.set(val)
                    slider.set(val)
                    self.update_image()
                except ValueError:
                    pass  # ignore bad input

            def on_slider_change(var, entry_widget, val):
                val = float(val)
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, f"{val:.2f}" if step < 1 else str(int(val)))
                var.set(val)
                self.update_image()

            entry.bind("<Return>", on_entry_change)

        # Checkbox for showing analysis
        ttk.Checkbutton(
            control_frame, text="Show analyzed", variable=self.params["show_analysis"],
            command=self.update_image
        ).pack(anchor="w", pady=5)

        # Checkbox for roundness
        ttk.Checkbutton(
            control_frame, text="Filter Roundness", variable=self.params["filter_roundness"],
            command=self.update_image
        ).pack(anchor="w", pady=5)

        # Sliders + Entry Boxes
        add_slider_with_entry("Roundness Threshold (%)",
                              self.params["filter_roundness_threshold"], 0.04, 0.99, step=0.05)
        add_slider_with_entry("Min Prop Area (%)",
                              self.params["filter_min_prop_area"], 0, 100, step=0.5)
        add_slider_with_entry("Tissue Hole Area",
                              self.params["filter_tissue_mask_holes_area"], 0, 20000, step=100)
        add_slider_with_entry("Small Object Min Size",
                              self.params["filter_tissue_mask_small_object_min_size"], 0, 1000, step=10)
        add_slider_with_entry("Tile Index",
                              self.params["tile_index"], 0, 9, step=1)

        ttk.Button(control_frame, text="Save", command=self.on_save).pack(anchor="w", pady=5)

        # Right panel: Image display
        self.image_panel = ttk.Label(main_frame)
        self.image_panel.pack(side="right", padx=10, pady=10)

    def update_analyzer_parameters(self):
        self.liver_fat_analyzer.set_params_from_dic(self.get_params())
        # TODO: I think this is a better replacement, but not sure
        # self.liver_fat_analyzer.filter_roundness = self.params['filter_roundness'].get()
        # self.liver_fat_analyzer.filter_roundness_threshold = self.params['filter_roundness_threshold'].get()
        # self.liver_fat_analyzer.filter_min_prop_area = self.params['filter_min_prop_area'].get()
        # self.liver_fat_analyzer.filter_tissue_mask_holes_area = self.params['filter_tissue_mask_holes_area'].get()
        # self.liver_fat_analyzer.filter_tissue_mask_small_object_min_size = self.params['filter_tissue_mask_small_object_min_size'].get()

    def on_save(self):
        self.callback(self.get_params())
        self.destroy()

    def on_cancel(self):
        self.callback(None)
        self.destroy()

    def update_image(self):
        self.update_analyzer_parameters()
        show_analysis = self.params['show_analysis'].get()
        index = self.params["tile_index"].get()
        tile = self.tiles[index].copy()
        tile_size = self.liver_fat_analyzer.tile_size
        if show_analysis:
            overlay_pil = self.liver_fat_analyzer._process_tile(x = tile['x'], y = tile['y'], return_tile_image=True)
        else:
            _, _, unproc_img = self.liver_fat_analyzer._unprocessed_tile(x=tile['x'], y=tile['y'])
            overlay_pil = Image.fromarray(unproc_img)
        overlay_pil = overlay_pil.resize((tile_size, tile_size), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(overlay_pil)

        self.image_panel.imgtk = imgtk
        self.image_panel.configure(image=imgtk)

    def get_params(self):
        return {'filter_roundness': self.params['filter_roundness'].get(),
                'filter_roundness_threshold': self.params['filter_roundness_threshold'].get(),
                'filter_min_prop_area': self.params['filter_min_prop_area'].get(),
                'filter_tissue_mask_holes_area': self.params['filter_tissue_mask_holes_area'].get(),
                'filter_tissue_mask_small_object_min_size': self.params['filter_tissue_mask_small_object_min_size'].get()
                }

# Run the interface
# if __name__ == "__main__":
#     slide_path = '14_07_24 hfd experiment/3 hfd control.mrxs'
#     root = tk.Tk()
#     app = FatAnalyzerParametersSelectionUI(root, slide_path)  # Replace with your image path
#     root.mainloop()
