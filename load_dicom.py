from datetime import datetime
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_dicom_images, process_image_data, unzip_dataset, apply_windowing, create_segmentation_dict, \
    save_to_csv


def visualize_dicom_slices(dicom_images, patient_dir, output_dir):
    """Visualize DICOM slices and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    window_level = 40
    window_width = 400
    for i, slice_img in enumerate(dicom_images):
        if i % 30 == 0:
            ct_array_windowed = apply_windowing(slice_img, window_level, window_width)
            plt.imshow(ct_array_windowed, cmap='gray')
            plt.title(f'Slice {i + 1}')
            plt.axis('off')
            plt.savefig(f'{output_dir}/{patient_dir}_slice_{i + 1}.png')
            plt.close()


def load_dataset(zip_path, save, viz):
    output_path = '/valohai/outputs/'

    base_path = unzip_dataset(zip_path, './unzipped_dataset')
    data_path = os.path.join(base_path, 'hcc_short/manifest-1643035385102/HCC-TACE-Seg/')

    data = {}
    for patient_dir in os.listdir(data_path):
        patient_path = Path(os.path.join(data_path, patient_dir))
        date_folders = [folder for folder in patient_path.glob('*') if folder.is_dir()]

        if not date_folders:
            continue

        # Sort date folders and select the earliest one (assumes MM-DD-YYYY format)
        earliest_date_folder = \
            sorted(date_folders, key=lambda x: datetime.strptime(x.name.split('-NA-')[0], '%m-%d-%Y'))[0]

        # Look for 'Recon 3 LIVER 3 PHASE' and 'Segmentation' folders inside the earliest date folder
        ct_path = None
        seg_path = None

        for content in earliest_date_folder.iterdir():
            if 'RECON 3' in content.name.upper() or '3 PHASE' in content.name.upper():
                ct_path = content
            if 'SEGMENTATION' in content.name.upper():
                seg_path = content

        if seg_path is None:
            print(f"Segmentation not found for patient {patient_dir}")

        if ct_path is None:
            candidate_folders = [
                content for content in earliest_date_folder.iterdir()
                if
                'SEGMENTATION' not in content.name.upper() and 'PRE' not in content.name.upper() and '.DS_STORE' not in content.name.upper()
            ]
            ct_path = max(candidate_folders, key=lambda x: int(str(x.name).split('.')[0]))

        print(f"ct_path is: {ct_path}")

        if ct_path and seg_path:
            # Load DICOM images
            dicom_images = load_dicom_images(ct_path)
            # Process the images (sorting and retrieving pixel data)
            processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing = process_image_data(dicom_images)

            if viz:
                # Visualize and save DICOM slices
                visualize_dicom_slices(dicom_pixel_array, patient_dir, output_dir=output_path)

            seg_file_path = seg_path / '1-1.dcm'
            if seg_file_path.exists():
                ds = pydicom.dcmread(seg_file_path)
                seg_dict = create_segmentation_dict(ds, processed_dicomset, dicom_pixel_array)

                data[patient_dir] = {
                    'ct_images': dicom_pixel_array,
                    'segmentation': seg_dict,
                    'slice_thickness': slice_thickness,
                    'pixel_spacing': pixel_spacing
                }
    if save:
        save_to_csv(data)
    return data


if __name__ == "__main__":
    data_path = "/valohai/inputs/hcc_dataset/hcc_short.zip"
    load_dataset(data_path, save=True, viz=True)
