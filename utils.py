import csv
import json
import os
import pydicom
import numpy as np
import zipfile


def unzip_dataset(zip_path, extract_to, verbose=False):
    """Unzip the dataset to a specified directory with verbose output."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if verbose:
            print(f"Extracting {zip_path} to {extract_to}...")
        for file in zip_ref.namelist():
            if verbose:
                print(f"Extracting {file}...")
            zip_ref.extract(file, extract_to)
    return extract_to


def load_dicom_images(path):
    """Load DICOM images from a specified directory."""
    # images = []
    img_dcmset = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(path, filename)
            ds = pydicom.dcmread(filepath)
            img_dcmset.append(ds)
            # images.append(ds.pixel_array)
    return img_dcmset


def process_image_data(img_dcmset):
    slice_thickness = img_dcmset[0].SliceThickness
    pixel_spacing = img_dcmset[0].PixelSpacing[0]

    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber is not None]

    # Making sure that there is only one acquisition
    acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]

    img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)

    return img_dcmset, img_pixelarray, slice_thickness, pixel_spacing


def apply_windowing(ct_array, window_level, window_width):
    """
    Apply windowing to CT image data.

    Parameters:
    - ct_array: numpy array of the CT image.
    - window_level: window level value.
    - window_width: window width value.

    Returns:
    - Windowed image array.
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    windowed_array = np.clip(ct_array, lower_bound, upper_bound)
    windowed_array = (windowed_array - lower_bound) / (upper_bound - lower_bound)  # Normalize to [0, 1]

    return np.clip(windowed_array, 0, 1)


def create_segmentation_dict(seg_dcm, img_dcm, img_pixel_array):
    segment_labels = {}
    if 'SegmentSequence' in seg_dcm:
        segment_sequence = seg_dcm.SegmentSequence
        for segment in segment_sequence:
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel.replace(" ", "").lower()
            segment_labels[segment_number] = segment_label

    mask_dcm = seg_dcm
    mask_pixelarray_messy = seg_dcm.pixel_array  # Pydicom's unordered PixelArray

    segmentation_dict = {
        f'seg_{label}': np.zeros(
            (img_pixel_array.shape[0], mask_pixelarray_messy.shape[1], mask_pixelarray_messy.shape[2]), dtype=np.uint8)
        for label in segment_labels.values()
    }

    first_slice_depth = img_dcm[0]['ImagePositionPatient'][2].real
    last_slice_depth = img_dcm[-1]['ImagePositionPatient'][2].real
    slice_increment = (last_slice_depth - first_slice_depth) / (len(img_dcm) - 1)
    for frame_idx, frame_info in enumerate(mask_dcm[0x52009230]):  # (5200 9230) -> Per-frame Functional Groups Sequence
        position = frame_info['PlanePositionSequence'][0]['ImagePositionPatient']
        slice_depth = position[2].real
        slice_idx = round((slice_depth - first_slice_depth) / slice_increment)

        segm_number = frame_info['SegmentIdentificationSequence'][0]['ReferencedSegmentNumber'].value

        if segm_number in segment_labels:
            segment_label = segment_labels[segm_number]
            if 0 <= slice_idx < segmentation_dict[f'seg_{segment_label}'].shape[0]:
                segmentation_dict[f'seg_{segment_label}'][slice_idx, :, :] = mask_pixelarray_messy[frame_idx, :,
                                                                             :].astype('int')

    return segmentation_dict


def save_to_csv(data, filename='/valohai/outputs/hcc_data.csv'):
    # Flatten the data into a list of dictionaries suitable for writing to a CSV
    flattened_data = []

    for patient_id, patient_data in data.items():
        # Base dictionary with patient-level information
        base_flat_dict = {
            'patient_id': patient_id,
            'slice_thickness': patient_data['slice_thickness'],
            'pixel_spacing': patient_data['pixel_spacing'],
        }

        # Flatten the ct_images and store it along with its shape
        ct_images = patient_data['ct_images']
        flat_ct_images = ct_images.flatten().tolist()  # Flatten the array to save as a list in CSV
        base_flat_dict['ct_images'] = flat_ct_images
        base_flat_dict['ct_images_shape'] = ct_images.shape  # Store the shape of the CT images

        # Now flatten the seg_dict
        segmentation = patient_data.get('segmentation', {})

        for organ, segmentation_array in segmentation.items():
            # Flatten the segmentation array and store it along with its shape
            flat_array = segmentation_array.flatten().tolist()  # Convert array to a flat list for CSV storage
            base_flat_dict[f'{organ}_segmentation'] = flat_array
            base_flat_dict[f'{organ}_shape'] = segmentation_array.shape  # Save the shape of the array

        flattened_data.append(base_flat_dict)

    # Make sure there's data to write
    if flattened_data:
        # Get all the keys from the first entry for the CSV header
        keys = flattened_data[0].keys()

        # Write the data to the CSV file
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(flattened_data)

        # Save metadata
        metadata = {"valohai.alias": "hcc_data"}
        metadata_path = '/valohai/outputs/hcc_data.csv.metadata.json'
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)

    else:
        print("No data to write to CSV.")


def parse_array_from_string(array_string, shape):
    """Convert a flattened string representation of an array to a numpy array and reshape it."""
    # Remove brackets and extra spaces
    cleaned_str = array_string.replace('[', '').replace(']', '').strip()

    # Convert the cleaned string to a numpy array (assuming the data is space-separated)
    array = np.fromstring(cleaned_str, sep=' ')

    # Reshape the array based on the provided shape
    reshaped_array = array.reshape(shape)

    return reshaped_array

def load_hcc_data(file_path):
    """Load hcc_data.csv containing CT images and segmentations."""
    # Increase CSV field size limit to avoid the field limit error
    csv.field_size_limit(sys.maxsize)

    data = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_id = row['patient_id']
            slice_thickness = float(row['slice_thickness'])
            pixel_spacing = float(row['pixel_spacing'])

            # Parse and reshape ct_images
            ct_images_shape = eval(row['ct_images_shape'])  # Convert shape string back to a tuple
            ct_images = parse_array_from_string(row['ct_images'], ct_images_shape)

            # Parse segmentation arrays and reshape using the saved shape
            liver_shape = eval(row['liver_shape'])  # Convert shape back to a tuple
            liver_segmentation = parse_array_from_string(row['liver_segmentation'], liver_shape)

            aorta_shape = eval(row['aorta_shape'])
            aorta_segmentation = parse_array_from_string(row['aorta_segmentation'], aorta_shape)

            portalvein_shape = eval(row['portalvein_shape'])
            portalvein_segmentation = parse_array_from_string(row['portalvein_segmentation'], portalvein_shape)

            mass_shape = eval(row['mass_shape'])
            mass_segmentation = parse_array_from_string(row['mass_segmentation'], mass_shape)

            # Store data for each patient
            data[patient_id] = {
                'ct_images': ct_images,
                'slice_thickness': slice_thickness,
                'pixel_spacing': pixel_spacing,
                'segmentation': {
                    'liver': liver_segmentation,
                    'aorta': aorta_segmentation,
                    'portalvein': portalvein_segmentation,
                    'mass': mass_segmentation
                }
            }
    return data
