import csv
import json
import SimpleITK as sitk
import os
import numpy as np
import radiomics
from radiomics import featureextractor

from utils import load_hcc_data


def extract_radiomics_features(patient_data, organ_label):
    """Extract radiomic features from CT images using the original segmentation mask."""
    ct_images = patient_data['ct_images']
    segmentation_mask = patient_data['segmentation'][organ_label]

    # Convert numpy arrays to SimpleITK images
    ct_image_sitk = sitk.GetImageFromArray(ct_images)
    segmentation_sitk = sitk.GetImageFromArray(segmentation_mask)

    # Set spacing and slice thickness
    spacing = (patient_data['pixel_spacing'], patient_data['pixel_spacing'], patient_data['slice_thickness'])
    ct_image_sitk.SetSpacing(spacing)
    segmentation_sitk.SetSpacing(spacing)

    # Define radiomics feature extraction settings
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'enableCExtensions': True
    }

    # Initialize radiomics feature extractor
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)

    try:
        features = extractor.execute(ct_image_sitk, segmentation_sitk)
        features = {key: value for key, value in features.items() if 'diagnostics' not in key}
    except Exception as e:
        print(f"Error extracting features for {organ_label}: {e}")
        features = {}

    return features


def save_radiomics_to_csv(features):
    """Save the extracted radiomics features to CSV."""
    if not features:
        print("No features to save.")
        return

    # Convert numpy arrays to native Python types
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            features[key] = value.item()  # Convert to scalar if array is 1-element

    pat_id = features['patient_id']
    output_filename = f'/valohai/outputs/{pat_id}_radiomics.csv'

    with open(output_filename, 'w', newline='') as output_file:
        fieldnames = features.keys()
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(features)  # Write a single row since `features` is a dictionary

    # Use execution id as dataset version name
    with open("/valohai/config/execution.json") as f:
        exec_id = json.load(f)["valohai.execution-id"]

    # Make it part of Valohai Dataset
    metadata = {
        "valohai.dataset-versions": [f"dataset://radiomics_extracted_from_dicom/{exec_id}"]
    }

    metadata_path = f'{output_filename}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

    print(f"Radiomics features saved to {output_filename}")


def main():
    # Load the data from hcc_data.csv
    hcc_data_path = "/valohai/inputs/prep_hcc_dataset/"
    for patient_file in os.listdir(hcc_data_path):
        patient_id = os.path.splitext(patient_file)[0]
        patient_data_path = os.path.join(hcc_data_path, patient_file)
        patient_data = load_hcc_data(patient_data_path)

        organ_label = 'seg_mass'  # Specify which organ to extract features for

        print(f"Extracting features for patient {patient_id}")

        # Extract features for the specified organ
        features = extract_radiomics_features(patient_data, organ_label)

        # Append additional patient information to the features
        feature_dict = {'patient_id': patient_id, 'organ': organ_label}
        feature_dict.update(features)

        # Save extracted features to CSV
        save_radiomics_to_csv(feature_dict)


if __name__ == "__main__":
    main()
