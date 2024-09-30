import os
import h5py
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


def load_hcc_data(file_path):
    """Load patient data from HDF5 format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Patient data file {file_path} not found")

    # Open the HDF5 file and load data
    with h5py.File(file_path, 'r') as hdf:
        print("Keys: %s" % hdf.keys())
        slice_thickness = hdf.attrs['slice_thickness']
        pixel_spacing = hdf.attrs['pixel_spacing']

        # Load ct_images
        ct_images = hdf['ct_images'][:]

        # Load segmentation masks (one for each organ)
        liver_segmentation = hdf.get('seg_liver', None)
        aorta_segmentation = hdf.get('seg_abdominalaorta', None)
        portalvein_segmentation = hdf.get('seg_portalvein', None)
        mass_segmentation = hdf.get('seg_mass', None)

        # Store data for this patient
        data = {
            'ct_images': ct_images,
            'slice_thickness': slice_thickness,
            'pixel_spacing': pixel_spacing,
            'segmentation': {
                'liver': liver_segmentation[:],
                'aorta': aorta_segmentation[:],
                'portalvein': portalvein_segmentation[:],
                'mass': mass_segmentation[:]
            }
        }

    return data