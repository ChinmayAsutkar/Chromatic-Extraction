#! /usr/bin/python3

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class RedactionTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Redaction Tool")
        
        # Store selected files
        self.object_files = []
        self.destination_file = None
        self.result_image = None
        self.preview_labels = []  # Add this to track preview labels
        
        # Configure root window to use full width
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
        # Create UI elements
        self.create_ui()
        
    def create_ui(self):
        # Add this near the start of create_ui
        style = ttk.Style()
        style.configure('Dark.TFrame', background='#333333')
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_columnconfigure(2, weight=1)  # Add column for redacted areas
        
        # Object files frame
        obj_frame = ttk.LabelFrame(main_container, text="Object Images", padding=10)
        obj_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Button(obj_frame, text="Add Object Image", 
                  command=self.add_object_file).grid(row=0, column=0, pady=5)
        
        self.obj_listbox = tk.Listbox(obj_frame, height=10, width=50)
        self.obj_listbox.grid(row=1, column=0, pady=5)
        
        # Object previews frame
        self.preview_frame = ttk.LabelFrame(obj_frame, text="Object Previews", padding=10)
        self.preview_frame.grid(row=2, column=0, pady=5, sticky="nsew")
        
        # Destination frame
        dest_frame = ttk.LabelFrame(main_container, text="Destination & Result", padding=10)
        dest_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        ttk.Button(dest_frame, text="Select Destination Image",
                  command=self.select_destination).grid(row=0, column=0, pady=5)
        
        self.dest_label = ttk.Label(dest_frame, text="No file selected")
        self.dest_label.grid(row=1, column=0, pady=5)
        
        # Destination preview
        self.dest_preview = ttk.Label(dest_frame)
        self.dest_preview.grid(row=2, column=0, pady=5)
        
        # Result preview
        self.result_preview = ttk.Label(dest_frame)
        self.result_preview.grid(row=3, column=0, pady=5)
        
        # Redacted areas frame
        self.redacted_frame = ttk.LabelFrame(main_container, text="Redacted Areas", padding=10)
        self.redacted_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        # Add scrollable frame for redacted areas
        self.redacted_canvas = tk.Canvas(self.redacted_frame)
        scrollbar = ttk.Scrollbar(self.redacted_frame, orient="vertical", command=self.redacted_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.redacted_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.redacted_canvas.configure(scrollregion=self.redacted_canvas.bbox("all"))
        )
        
        self.redacted_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.redacted_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.redacted_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons frame
        button_frame = ttk.Frame(main_container)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Redact Black", 
                  command=lambda: self.redact_images("black")).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Redact Blur",
                  command=lambda: self.redact_images("blur")).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Result",
                  command=self.save_result).grid(row=0, column=2, padx=5)
        
    def resize_image_for_preview(self, image, max_size=300):
        # Convert cv2 image to PIL Image
        if isinstance(image, np.ndarray):
            # Handle RGBA images
            if image.shape[2] == 4:
                image = Image.fromarray(image, 'RGBA')
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
        # Calculate new size maintaining aspect ratio
        ratio = min(max_size/image.size[0], max_size/image.size[1])
        new_size = tuple(int(dim * ratio) for dim in image.size)
        
        return ImageTk.PhotoImage(image.resize(new_size, Image.Resampling.LANCZOS))

    def create_delete_button(self, index, container):
        delete_btn = ttk.Button(container, text="🗑", width=3,
                              command=lambda idx=index: self.remove_object(idx))
        return delete_btn

    def remove_object(self, index):
        # Remove from data structures
        self.object_files.pop(index)
        self.obj_listbox.delete(index)
        
        # Clear all previews
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self.preview_labels.clear()
        
        # Recreate all previews
        for i, filename in enumerate(self.object_files):
            img = Image.open(filename)
            photo = self.resize_image_for_preview(img)
            
            # Create container frame for preview and delete button
            container = ttk.Frame(self.preview_frame)
            container.grid(row=i, column=0, pady=5, sticky="ew")
            
            # Add preview label
            preview_label = ttk.Label(container, image=photo)
            preview_label.image = photo
            preview_label.grid(row=0, column=0, padx=(0, 5))
            
            # Add delete button with container as parent
            delete_btn = self.create_delete_button(i, container)
            delete_btn.grid(row=0, column=1)
            
            self.preview_labels.append(preview_label)

    def add_object_file(self):
        if len(self.object_files) >= 10:
            tk.messagebox.showwarning("Warning", "Maximum 10 object files allowed")
            return
            
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if filename:
            self.object_files.append(filename)
            self.obj_listbox.insert(tk.END, filename.split("/")[-1])
            
            # Create container frame for preview and delete button
            container = ttk.Frame(self.preview_frame)
            container.grid(row=len(self.object_files)-1, column=0, pady=5, sticky="ew")
            
            # Add preview
            img = Image.open(filename)
            photo = self.resize_image_for_preview(img)
            preview_label = ttk.Label(container, image=photo)
            preview_label.image = photo
            preview_label.grid(row=0, column=0, padx=(0, 5))
            
            # Add delete button with container as parent
            delete_btn = self.create_delete_button(len(self.object_files)-1, container)
            delete_btn.grid(row=0, column=1)
            
            self.preview_labels.append(preview_label)

    def select_destination(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if filename:
            self.destination_file = filename
            self.dest_label.config(text=filename.split("/")[-1])
            
            # Update preview
            img = Image.open(filename)
            photo = self.resize_image_for_preview(img)
            self.dest_preview.configure(image=photo)
            self.dest_preview.image = photo

    def extract_redacted_areas(self, image, corners_list):
        print(f"\nExtracting {len(corners_list)} redacted areas...")
        extracted_areas = []
        for i, corners in enumerate(corners_list):
            print(f"Processing area {i+1}...")
            try:
                # Create mask for the exact polygon
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [corners], 255)
                
                # Get bounding rectangle for the region (to minimize image size)
                x, y, w, h = cv2.boundingRect(corners)
                print(f"Area {i+1} bounds: x={x}, y={y}, w={w}, h={h}")
                
                # Ensure bounds are within image dimensions
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    print(f"Skipping area {i+1} - invalid dimensions")
                    continue
                
                # Create transparent background
                area = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA image
                
                # Copy only the pixels inside the polygon
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    print(f"Skipping area {i+1} - empty ROI")
                    continue
                    
                roi_mask = mask[y:y+h, x:x+w]
                
                # Convert BGR to RGB and copy
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                area[..., :3][roi_mask == 255] = roi_rgb[roi_mask == 255]
                area[..., 3][roi_mask == 255] = 255
                
                print(f"Area {i+1} shape: {area.shape}")
                extracted_areas.append(area)
                
            except Exception as e:
                print(f"Error processing area {i+1}: {str(e)}")
                continue
            
        print(f"Extracted {len(extracted_areas)} areas")
        return extracted_areas

    def find_matches(self, img1, img2):
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        if des1 is None or des2 is None:
            print("No descriptors found")
            return None
            
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=150)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        print(f"Found {len(good)} good matches")
                
        if len(good) < 10:  # Back to original threshold of 10
            print("Not enough good matches")
            return None
            
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("No homography found")
            return None
            
        # Count inliers
        inliers = np.sum(mask)
        print(f"Found {inliers} inliers")
        if inliers < 10:  # Back to original threshold of 10
            print("Not enough inliers")
            return None
            
        # Get corners of object image
        h, w = img1_gray.shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners to destination image coordinates
        transformed_corners = cv2.perspectiveTransform(corners, H)
        print("Match found successfully")
        
        return np.int32(transformed_corners)

    def show_redacted_areas(self, areas):
        print(f"\nShowing {len(areas)} redacted areas...")
        # Clear previous areas
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Show each extracted area
        for i, area in enumerate(areas):
            print(f"Displaying area {i+1}, shape: {area.shape}")
            # Create container for each area
            container = ttk.Frame(self.scrollable_frame)
            container.grid(row=i, column=0, pady=5, sticky="ew")
            
            # Add label
            ttk.Label(container, text=f"Redacted Area {i+1}").grid(row=0, column=0, pady=2)
            
            # Create a dark background frame
            bg_frame = ttk.Frame(container, style='Dark.TFrame')
            bg_frame.grid(row=1, column=0, pady=2)
            
            try:
                # Convert to PIL Image first
                pil_image = Image.fromarray(area)
                # Add preview on dark background
                photo = self.resize_image_for_preview(pil_image, max_size=200)
                preview = ttk.Label(bg_frame, image=photo)
                preview.image = photo
                preview.grid(row=0, column=0, pady=2)
                print(f"Successfully displayed area {i+1}")
            except Exception as e:
                print(f"Error displaying area {i+1}: {str(e)}")
        
        # Force canvas update
        self.scrollable_frame.update_idletasks()
        self.redacted_canvas.update_idletasks()
        print("Finished showing redacted areas")

    def redact_images(self, mode="black"):
        if not self.object_files or not self.destination_file:
            tk.messagebox.showerror("Error", "Please select both object and destination images")
            return
            
        print("\nStarting redaction process...")
        # Read destination image
        dest_img = cv2.imread(self.destination_file)
        if dest_img is None:
            tk.messagebox.showerror("Error", "Could not read destination image")
            return
            
        self.result_image = dest_img.copy()
        original_img = dest_img.copy()  # Keep a clean copy for extraction
        
        # Store corners for extraction
        all_corners = []
        matches_found = 0
        total_objects = len(self.object_files)
        
        # Process each object image
        for i, obj_file in enumerate(self.object_files):
            print(f"\nProcessing object {i+1}/{total_objects}: {obj_file}")
            obj_img = cv2.imread(obj_file)
            if obj_img is None:
                print(f"Could not read object image: {obj_file}")
                continue
                
            # Find matches and get corners
            corners = self.find_matches(obj_img, self.result_image)
            
            if corners is not None:
                matches_found += 1
                all_corners.append(corners)
                print(f"Found match {matches_found}, corners shape: {corners.shape}")
                # Create mask for the matched region
                mask = np.zeros(self.result_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [corners], 255)
                
                if mode == "black":
                    # Apply black redaction
                    self.result_image[mask == 255] = [0, 0, 0]
                else:  # blur mode
                    # Create blurred version of the image
                    blurred = cv2.GaussianBlur(self.result_image, (99, 99), 30)
                    # Copy only the blurred regions we want
                    self.result_image = np.where(mask[:, :, np.newaxis] == 255, 
                                               blurred, 
                                               self.result_image)
        
        print(f"\nFound {matches_found} out of {total_objects} matches")
        
        # Show status message
        if matches_found == 0:
            tk.messagebox.showinfo("Result", "No matching areas found in the destination image")
            return
        elif matches_found < total_objects:
            tk.messagebox.showinfo("Result", f"Found {matches_found} out of {total_objects} matching areas")
        
        # Show the redacted result first
        print("\nUpdating result preview...")
        photo = self.resize_image_for_preview(self.result_image)
        self.result_preview.configure(image=photo)
        self.result_preview.image = photo
        
        # Then extract and show redacted areas
        if all_corners:
            print("\nExtracting and showing redacted areas...")
            extracted_areas = self.extract_redacted_areas(original_img, all_corners)
            self.show_redacted_areas(extracted_areas)
        
        print("Redaction process complete")

    def save_result(self):
        if self.result_image is None:
            tk.messagebox.showerror("Error", "No redacted image to save")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile="redacted_" + self.destination_file.split("/")[-1]
        )
        
        if output_path:
            cv2.imwrite(output_path, self.result_image)
            tk.messagebox.showinfo("Success", f"Redacted image saved as {output_path}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RedactionTool()
    app.run()
