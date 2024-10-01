
# DICOM File Processing with Valohai

This repository demonstrates how to efficiently process DICOM files for medical imaging tasks using Valohai. The example focuses on handling healthcare-related data, specifically hepatocellular carcinoma (HCC) imaging datasets. Through Valohai's robust workflow orchestration, we automate the loading and processing of DICOM files, extract valuable radiomic features, and visualize the data with a Jupyter Notebook.

## Project Overview

This project contains a two-step pipeline defined in the `valohai.yaml` file:

1. **Load DICOM Data**: This step loads a dataset of DICOM files, decompresses it, and prepares it for further analysis. We utilize sample HCC imaging data from a publicly available dataset.

2. **Extract Radiomic Features**: Once the DICOM files are loaded, this step extracts relevant radiomic features, which are useful in clinical decision-making and research for understanding tumor characteristics.

### Key Features:
- **Automated Pipeline**: The pipeline ensures a smooth transition from loading raw DICOM data to extracting radiomic features.
- **Scalability with Valohai**: This project leverages Valohai's cloud capabilities to handle larger datasets and more complex workflows with ease.
- **Visualization**: In addition to the pipeline steps, the repository includes a Jupyter Notebook that demonstrates how to visualize DICOM data slices across different planes (axial, coronal, sagittal) and even generate animated GIFs for better data understanding.

## Visualization Notebook

The Jupyter Notebook provided in this repository showcases how to visualize DICOM images interactively. The visualization includes 2D slices from different planes (axial, coronal, sagittal) for specific patients. Additionally, it can create 3D representations by generating rotating animations of the DICOM data.

The notebook demonstrates:
- Loading preprocessed DICOM files.
- Visualizing axial, coronal, and sagittal slices.
- Creating a rotating animated GIF to display the volumetric data in 3D.

The notebook is a helpful tool for researchers and clinicians to explore medical imaging data visually and interactively.

## Running on Valohai

### Configure the repository:

To run your code on Valohai using the terminal, follow these steps:

1. Install Valohai on your machine by running the following command:

    ```bash
    pip install valohai-cli valohai-utils
    ```

2. Log in to Valohai from the terminal using the command:

    ```bash
    vh login
    ```

3. Create a project for your Valohai workflow.

    Start by creating a directory for your project:

    ```bash
    mkdir valohai-dicom-example
    cd valohai-dicom-example
    ```

    Then, create the Valohai project:

    ```bash
    vh project create
    ```

4. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/valohai/dicom-example.git .
    ```

Congratulations! You have successfully cloned the repository, and you can now modify the code and run it using Valohai.

### Running Executions:

To run individual steps, execute the following command:

```bash
vh execution run <step-name> --adhoc
```

For example, to run the `load_dicom` step, use the command:

```bash
vh execution run load_dicom --adhoc
```

### Running Pipelines:

To run the full pipeline, use the following command:

```bash
vh pipeline run hcc_pipeline --adhoc
```

This will execute the entire workflow, from loading the DICOM data to extracting the radiomic features.
