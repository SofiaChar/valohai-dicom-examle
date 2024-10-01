from datetime import datetime
import os
import numpy as np
import h5py
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_dicom_images, process_image_data, unzip_dataset, apply_windowing, create_segmentation_dict
from visualise import visualize_dicom_slices, visualize_dicom


def save_patient_data_to_hdf5(patient_id, patient_data, output_dir='/valohai/outputs/'):
    """Save each patient's data incrementally to an HDF5 file to avoid memory overload."""
    output_file = os.path.join(output_dir, f"{patient_id}.hdf5")

    with h5py.File(output_file, 'w') as hdf:
        # Save basic patient-level metadata
        hdf.attrs['patient_id'] = patient_id
        hdf.attrs['slice_thickness'] = patient_data['slice_thickness']
        hdf.attrs['pixel_spacing'] = patient_data['pixel_spacing']

        # Save ct_images as a dataset
        hdf.create_dataset('ct_images', data=patient_data['ct_images'])

        # Save each segmentation as a separate dataset
        segmentation = patient_data['segmentation']
        for organ, segmentation_array in segmentation.items():
            hdf.create_dataset(f'{organ}', data=segmentation_array)

    print(f"Data for patient {patient_id} saved to {output_file}")


def load_dataset(zip_path, output_dir, viz=False):
    base_path = unzip_dataset(zip_path, './unzipped_dataset')
    data_path = os.path.join(base_path, 'hcc_short/manifest-1643035385102/HCC-TACE-Seg/')

    for patient_dir in os.listdir(data_path):
        process_patient_data(patient_dir, data_path, output_dir, viz)


def process_patient_data(patient_dir, data_path, output_dir, viz):
    patient_path = Path(os.path.join(data_path, patient_dir))
    date_folders = get_date_folders(patient_path)

    if not date_folders:
        return

    earliest_date_folder = get_earliest_date_folder(date_folders)
    ct_path, seg_path = find_ct_and_seg_paths(earliest_date_folder)

    if not (ct_path and seg_path):
        print(f"Skipping patient {patient_dir}: No segmentation or CT data found")
        return

    processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing = process_dicom_images(ct_path)

    process_segmentation_data(seg_path, processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing,
                              patient_dir, output_dir)

    if viz:
        visualize_dicom_slices(dicom_pixel_array, patient_dir, output_dir)


def get_date_folders(patient_path):
    return [folder for folder in patient_path.glob('*') if folder.is_dir()]


def get_earliest_date_folder(date_folders):
    return sorted(date_folders, key=lambda x: datetime.strptime(x.name.split('-NA-')[0], '%m-%d-%Y'))[0]


def find_ct_and_seg_paths(earliest_date_folder):
    ct_path, seg_path = None, None
    for content in earliest_date_folder.iterdir():
        if 'RECON 3' in content.name.upper() or '3 PHASE' in content.name.upper():
            ct_path = content
        if 'SEGMENTATION' in content.name.upper():
            seg_path = content
    return ct_path, seg_path


def process_dicom_images(ct_path):
    dicom_images = load_dicom_images(ct_path)
    processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing = process_image_data(dicom_images)
    return processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing


def process_segmentation_data(seg_path, processed_dicomset, dicom_pixel_array, slice_thickness, pixel_spacing,
                              patient_dir, output_dir):
    seg_file_path = seg_path / '1-1.dcm'
    if seg_file_path.exists():
        ds = pydicom.dcmread(seg_file_path)
        seg_dict = create_segmentation_dict(ds, processed_dicomset, dicom_pixel_array)
        if patient_dir == 'HCC_003':  # error in the dataset. the segmentation is fliped
            flipped_seg_dict = {key: np.flip(value, axis=1) for key, value in seg_dict.items()}
            seg_dict = {key: np.flip(value, axis=2) for key, value in flipped_seg_dict.items()}

        patient_data = {
            'patient_id': patient_dir,
            'ct_images': dicom_pixel_array,
            'segmentation': seg_dict,
            'slice_thickness': slice_thickness,
            'pixel_spacing': pixel_spacing
        }

        visualize_dicom(patient_data, slice_indices=[10, 20, 30])
        save_patient_data_to_hdf5(patient_dir, patient_data, output_dir)


if __name__ == "__main__":
    data_path = "/valohai/inputs/hcc_dataset/hcc_short.zip"
    output_dir = "/valohai/outputs"
    load_dataset(data_path, output_dir, viz=True)
