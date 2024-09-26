import argparse
import csv
import json
import SimpleITK as sitk
from tqdm import tqdm
import os
import numpy as np
import radiomics
from radiomics import featureextractor

import sys
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


def save_radiomics_to_csv(all_features, output_filename='/valohai/outputs/hcc_radiomics.csv'):
    """Save the extracted radiomics features to CSV."""
    if not all_features:
        print("No features to save.")
        return

    with open(output_filename, 'w', newline='') as output_file:
        fieldnames = all_features[0].keys()
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    metadata = {"valohai.alias": "hcc_radiomics_extracted"}

    metadata_path = f'{output_filename}.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

    print(f"Radiomics features saved to {output_filename}")


def main():
    # Load the data from hcc_data.csv
    hcc_data_file = "/valohai/inputs/prep_hcc_dataset/hcc_data.csv"
    data = load_hcc_data(hcc_data_file)

    all_features = []
    organ_label = 'mass'  # Specify which organ to extract features for

    # Iterate over each patient and extract features
    for patient_id, patient_data in tqdm(data.items()):
        print(f"Extracting features for patient {patient_id}")

        # Extract features for the specified organ
        features = extract_radiomics_features(patient_data, organ_label)

        # Append additional patient information to the features
        feature_dict = {'patient_id': patient_id, 'organ': organ_label}
        feature_dict.update(features)
        all_features.append(feature_dict)

    # Save extracted features to CSV
    save_radiomics_to_csv(all_features)


if __name__ == "__main__":
    main()
