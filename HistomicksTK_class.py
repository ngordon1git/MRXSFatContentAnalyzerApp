import numpy as np
import openslide
import cv2
import os
import pandas as pd
from skimage import morphology, measure
from typing import List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt


class LiverFatAnalyzer:
    def __init__(self, slide_path: str, tile_size: int = 1024, level: int = 0, save_tiles_image_percentage = 10,
                 min_tissue_percent: float = 5.0, slide_bounds = None,
                 output_dir: Optional[str] = None):

        self.slide_path = slide_path
        self.tile_size = tile_size
        self.save_tiles_image_percentage = save_tiles_image_percentage
        self.slide_bounds = slide_bounds # if not None, this restricts the part of the slide being analyzed.
        self.level = level
        self.min_tissue_percent = min_tissue_percent

        self.output_dir = output_dir or os.path.splitext(os.path.basename(slide_path))[0] + "_tiles"
        os.makedirs(self.output_dir, exist_ok=True)

        # Parameters for filtering fat vacuoles
        self.filter_roundness = True
        self.filter_min_prop_area = 50
        self.filter_tissue_mask_holes_area = 15000
        self.filter_tissue_mask_small_object_min_size = 500
        self.filter_roundness_threshold = 0.75

        self.results = []
        self.slide = openslide.OpenSlide(slide_path)
        self.slide_dims = self.slide.level_dimensions[self.level]

    def get_tiles(self, only_tiles_to_save = False):
        if self.slide_bounds is None:
            width, height = self.slide_dims
            x0,y0 = 0,0
        else:
            if type(self.slide_bounds) is tuple:
                x0,y0,width,height = self.slide_bounds
            elif type(self.slide_bounds) is dict:
                x0, y0, width, height = self.slide_bounds['x0'],self.slide_bounds['y0'],self.slide_bounds['width'],self.slide_bounds['height']
        # decide which tiles to save
        n_tiles = len(range(y0, y0 + height, self.tile_size)) * len(range(x0, x0 + width, self.tile_size))
        n_tiles_to_save = int(n_tiles * self.save_tiles_image_percentage) if  (type(self.save_tiles_image_percentage) is float and self.save_tiles_image_percentage <= 1.0 and self.save_tiles_image_percentage >= 0.0) else int(self.save_tiles_image_percentage)
        tiles_to_save = np.linspace(0,n_tiles, n_tiles_to_save , endpoint = False,dtype = int)
        i, tiles = 0, []
        for y in range(y0, y0 + height, self.tile_size):
            for x in range(x0, x0 + width, self.tile_size):
                if not only_tiles_to_save or i in tiles_to_save:
                    tiles.append({'x':x, 'y':y, 'save_tile':i in tiles_to_save})
                i += 1
        return tuple(tiles)
    def process_slide(self):
        """Process the entire slide tile-by-tile and collect fat analysis data with visualization."""
        tiles = self.get_tiles()
        for i, tile in enumerate(tiles):
            tile_result = self._process_tile(tile['x'], tile['y'], save_tile_image=tile['save_tile'])
            if tile_result:
                self.results.append(tile_result)
    def _unprocessed_tile(self, x: int, y: int):
        tile_w = min(self.tile_size, self.slide_dims[0] - x)
        tile_h = min(self.tile_size, self.slide_dims[1] - y)
        region = self.slide.read_region((x, y), self.level, (tile_w, tile_h)).convert('RGB')
        return  tile_w, tile_h, np.array(region)
    def _process_tile(self, x: int, y: int, save_tile_image: bool = False, return_tile_image: bool = False) -> Optional[dict]:
        tile_w, tile_h, image_rgb = self._unprocessed_tile(x,y)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Tissue mask
        _, tissue_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        tissue_mask = morphology.remove_small_objects(tissue_mask.astype(bool), min_size=self.filter_tissue_mask_small_object_min_size)
        tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=self.filter_tissue_mask_holes_area)

        tissue_area = np.count_nonzero(tissue_mask)
        total_area = tile_w * tile_h
        tissue_percent = (tissue_area / total_area) * 100

        if tissue_percent < self.min_tissue_percent:
            return None

        # Fat detection
        _, fat_mask_raw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        fat_mask = np.logical_and(fat_mask_raw, tissue_mask)

        # Optional: Morphological closing to merge nearby fat droplets
        fat_mask = morphology.closing(fat_mask, morphology.disk(3))

        # Roundness filter using regionprops
        labeled = measure.label(fat_mask)
        props = measure.regionprops(labeled)

        final_mask = fat_mask
        if self.filter_roundness:
            final_mask = np.zeros_like(fat_mask, dtype=bool)
            for prop in props:
                if prop.area < self.filter_min_prop_area:
                    continue
                if prop.eccentricity < self.filter_roundness_threshold:
                    final_mask[labeled == prop.label] = True

        fat_area = np.count_nonzero(final_mask)
        fat_percent = (fat_area / tissue_area) * 100 if tissue_area > 0 else 0

        # Save visual output
        if save_tile_image or return_tile_image:
            non_tissue_overlay = np.logical_not(tissue_mask)
            overlay = image_rgb.copy()
            overlay[final_mask] = [255, 0, 0]
            overlay[non_tissue_overlay] = [0,0,0] # Paint non-tissue areas black
            out_img = Image.fromarray(overlay)
            if return_tile_image: return out_img
            out_img.save(os.path.join(self.output_dir, f"tile_{x}_{y}.png"))

        return {
            'tile_x': x,
            'tile_y': y,
            'tissue_area': tissue_area,
            'fat_area': fat_area,
            'fat_percent': fat_percent,
            'tissue_percent': tissue_percent
        }

    def export_csv(self, output_csv: str):
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
    def export_fat_content_histogram(self, output_path: str, bins = 20):
        """
        Save a histogram of fat percentage across tiles.

        Args:
            output_path (str): File path to save the histogram image.
            bins (int): Number of histogram bins.
        """
        if not hasattr(self, 'results') or not self.results:
            print("No tile data available. Run process_slide() first.")
            return

        fat_values = [res['fat_percent'] for res in self.results if res is not None]

        plt.figure(figsize=(8, 5))
        plt.hist(fat_values, bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel("Fat Percentage per Tile (%)")
        plt.ylabel("Number of Tiles")
        plt.title("Distribution of Fat Content Across Tiles")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"Saved fat percentage histogram to: {output_path}")

    def summarize(self):
        if not self.results:
            print("No valid tiles with tissue detected.")
            return
        df = pd.DataFrame(self.results)
        # Compute total fat stats
        total_tissue = df['tissue_area'].sum()
        total_fat = df['fat_area'].sum()
        overall_fat_percent = (total_fat / total_tissue) * 100 if total_tissue > 0 else 0

        # Compute fat percent per tile, then std
        df['tile_fat_percent'] = (df['fat_area'] / df['tissue_area']) * 100
        std_fat_percent = df['tile_fat_percent'].std()

        print(f"Overall estimated fat content: {overall_fat_percent:.2f}% (± {std_fat_percent:.2f})")
        results = {'total_tissue':total_tissue,
                   'total_fat':total_fat,
                   'fat_content':overall_fat_percent,
                   'fat_content_std':std_fat_percent
                   }
        return results

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def display_thumbnail_for_bounds_selection(self, thumbnail_size = (2000,2000), resize_bounds = True, save_selected_thumbnail = True):
        # ------- Display thumbnail -------
        thumb = self.slide.get_thumbnail(thumbnail_size)
        bounds = self.get_visible_image_bounds(image = thumb, save_selected_thumbnail = save_selected_thumbnail)
        if resize_bounds:
            thumbnail_size_ratio = max(self.slide_dims) // max(thumb.size)
            self.slide_bounds = tuple([n * thumbnail_size_ratio for n in bounds])
        return (self.slide_bounds if resize_bounds else bounds)


    def get_visible_image_bounds(self, image: np.ndarray, save_selected_thumbnail:bool):
        """
        Display an image using matplotlib and return the top-left pixel (x, y)
        and visible width and height (in pixels) after the figure is closed.

        Args:
            image (np.ndarray): Image to display (H x W x 3 or 2D array).

        Returns:
            (x, y, w, h): Top-left pixel coordinates and visible width and height.
        """
        if hasattr(image, 'shape'):
            image = image
        else:
            # Assume PIL Image
            image = np.array(image)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title("Close the window to get visible bounds.")
        plt.show()

        # Get current visible area in data coordinates (pixels)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Convert from float and clip to image bounds
        x0 = int(max(min(xlim[0], xlim[1]), 0))
        x1 = int(min(max(xlim[0], xlim[1]), image.shape[1]))
        y0 = int(max(min(ylim[0], ylim[1]), 0))
        y1 = int(min(max(ylim[0], ylim[1]), image.shape[0]))

        width = x1 - x0
        height = y1 - y0

        if save_selected_thumbnail:
            # Extract and save visible region
            thumbnail = image[y0:y1, x0:x1]
            Image.fromarray(thumbnail).save(os.path.join(self.output_dir, f"thumbnail_selected_region.png"))

        return (x0, y0, width, height)

    def set_params_from_dic(self, d):
        self.filter_roundness = d['filter_roundness']
        self.filter_min_prop_area = d['filter_min_prop_area']
        self.filter_tissue_mask_holes_area = d['filter_tissue_mask_holes_area']
        self.filter_tissue_mask_small_object_min_size = d['filter_tissue_mask_small_object_min_size']
        self.filter_roundness_threshold = d['filter_roundness_threshold']

    @staticmethod
    def process_folder(input_folder: str, tile_size: int = 1024, level: int = 0,
                       min_tissue_percent: float = 5.0):
        """Process all MRXS files in a folder."""
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.mrxs'):
                full_path = os.path.join(input_folder, filename)
                print(f"Processing {filename}...")
                analyzer = LiverFatAnalyzer(
                    slide_path=full_path,
                    tile_size=tile_size,
                    level=level,
                    min_tissue_percent=min_tissue_percent,
                    output_dir=os.path.join(input_folder, f"{os.path.splitext(filename)[0]}_tiles")
                )
                analyzer.process_slide()
                csv_path = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}_fat_results.csv")
                analyzer.export_csv(csv_path)
                analyzer.summarize()

if __name__ == '__main__':
    slide_path = '14_07_24 hfd experiment/3 hfd control.mrxs'
    analyzer = LiverFatAnalyzer(slide_path)
    analyzer.display_thumbnail_for_bounds_selection()
    analyzer.process_slide()
    analyzer.export_csv('slide_fat_results.csv')
    analyzer.export_fat_content_histogram('fat_content_histogram.png')
    analyzer.summarize()
