import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import imageio
import os


# Normalize the image for display
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


# Maximum intensity projection function
def maximum_intensity_projection(img, axis):
    return np.max(img, axis=axis)


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


# Function to create GIF from rotated DICOM slices with segmentation overlays
def create_gif(img_pixelarray, seg_3d, slice_thickness, pixel_spacing, output_dir, patient_id, n_frames=30, alpha=0.3):
    """
    Create a rotating Maximum Intensity Projection (MIP) GIF from 3D DICOM images and segmentation masks.

    Args:
    - img_pixelarray: 3D array of DICOM images.
    - seg_3d: 3D array of segmentation masks.
    - slice_thickness: Thickness of each slice in mm.
    - pixel_spacing: Pixel spacing in mm.
    - output_dir: Directory to save the resulting GIF.
    - n_frames: Number of frames in the GIF animation.
    - alpha: Transparency value for the segmentation mask overlay.
    """
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Liver (red)
        [0.0, 1.0, 0.0, 1.0],  # Tumor (green)
        [0.0, 0.0, 1.0, 1.0],  # Vein (blue)
        [1.0, 1.0, 0.0, 1.0]  # Aorta (yellow)
    ]

    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        # Rotate the DICOM images and segmentation masks
        rotated_img = rotate(img_pixelarray, angle, axes=(1, 2), reshape=False)
        rotated_seg = rotate(seg_3d, angle, axes=(1, 2), reshape=False)

        # Perform Maximum Intensity Projection (MIP)
        mip_img = maximum_intensity_projection(rotated_img, axis=1)
        mip_seg = maximum_intensity_projection(rotated_seg, axis=1)

        # Normalize the image for better visualization
        mip_img_norm = normalize(mip_img)

        # Apply colormap to the MIP image
        cmap_bone = plt.get_cmap('bone')
        mip_img_colored = cmap_bone(mip_img_norm)

        # Extract the RGB channels from the colored MIP image
        combined_image = mip_img_colored[..., :3]

        # Overlay segmentation masks using different colors
        for idx, color in enumerate(colors):
            mask = (mip_seg == (idx + 1)).astype(float)
            mask_colored = color[:3] * mask[..., np.newaxis]
            combined_image = combined_image * (1 - alpha * mask[..., np.newaxis]) + mask_colored * alpha

        # Convert the combined image to uint8 for GIF creation
        combined_image_uint8 = (combined_image * 255).astype(np.uint8)
        frames.append(combined_image_uint8)

        # Optionally, save each frame as a PNG image for debugging/inspection
        plt.imshow(combined_image_uint8, cmap='prism', aspect=slice_thickness / pixel_spacing)
        plt.axis('off')
        plt.savefig(f'{output_dir}/frame_{i:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    # Save frames as an animated GIF
    gif_output_path = os.path.join(output_dir, f'{patient_id}_rotating_mip_animation.gif')
    imageio.mimsave(gif_output_path, [imageio.v2.imread(f'{output_dir}/frame_{i:03d}.png') for i in range(n_frames)],
                    fps=10)
    print(f"Animation saved as '{gif_output_path}'")


# Apply windowing for better visualization
def apply_windowing(image, window_level, window_width):
    """Applies windowing to the image based on window level and width."""
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2
    windowed_image = np.clip(image, min_intensity, max_intensity)
    return (windowed_image - min_intensity) / (max_intensity - min_intensity)


# Function to visualize and save DICOM slices with multiple segmentations
def visualize_dicom(dicom_images, seg_dict, patient_id, output_dir, slice_thickness, pixel_spacing,
                           plane='axial', alpha=0.5, slice_indices=None):
    """
    Visualize DICOM slices with segmentation masks and save them to the output directory.

    Args:
    - dicom_images: 3D array of DICOM slices.
    - seg_dict: Dictionary of segmentation masks.
    - patient_id: ID of the patient.
    - output_dir: Directory to save the images.
    - slice_thickness: Thickness of each slice in mm.
    - pixel_spacing: Pixel spacing in mm.
    - plane: Choose between 'axial', 'sagittal', and 'coronal'.
    - alpha: Transparency value for the segmentation mask overlay (0 = transparent, 1 = opaque).
    - slice_indices: List of specific slices to visualize. If None, defaults to all slices.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for different segmentation masks
    SEG_COLORS = {
        'seg_liver': [1, 0, 0],  # Red for liver
        'seg_mass': [0, 1, 0],  # Green for mass
        'seg_portalvein': [0, 0, 1],  # Blue for portal vein
        'seg_abdominalaorta': [1, 1, 0]  # Yellow for abdominal aorta
    }

    # Set default slice_indices if none are provided
    if slice_indices is None:
        if plane == 'axial':
            slice_indices = range(dicom_images.shape[0])  # Axial: slice along the first axis
        elif plane == 'sagittal':
            slice_indices = range(dicom_images.shape[1])  # Sagittal: slice along the second axis
        elif plane == 'coronal':
            slice_indices = range(dicom_images.shape[2])  # Coronal: slice along the third axis
        else:
            raise ValueError("Invalid plane selection. Choose from 'axial', 'sagittal', or 'coronal'.")

    # Set aspect ratio for non-axial slices
    if plane in ['sagittal', 'coronal']:
        aspect_ratio = slice_thickness / pixel_spacing  # Adjust aspect ratio for sagittal and coronal planes
    else:
        aspect_ratio = 1  # No need to adjust aspect ratio for axial

    # Window level and width for CT images
    window_level = 40
    window_width = 400

    # Loop through specified slices
    for i in slice_indices:
        if plane == 'axial':
            slice_img = dicom_images[i, :, :]  # Axial slice
        elif plane == 'sagittal':
            slice_img = dicom_images[:, i, :]  # Sagittal slice
        elif plane == 'coronal':
            slice_img = dicom_images[:, :, i]  # Coronal slice

        # Apply windowing to the DICOM slice
        ct_array_windowed = apply_windowing(slice_img, window_level, window_width)

        # Create an RGB image from the CT slice for overlaying masks
        ct_rgb = np.repeat(ct_array_windowed[..., np.newaxis], 3, axis=-1)

        # Create an empty mask overlay (for combining all masks)
        combined_seg_rgb = np.zeros_like(ct_rgb)

        # Overlay segmentation masks using different colors
        for seg_name, seg_mask in seg_dict.items():
            if plane == 'axial':
                seg_slice = seg_mask[i, :, :]  # Axial slice
            elif plane == 'sagittal':
                seg_slice = seg_mask[:, i, :]  # Sagittal slice
            elif plane == 'coronal':
                seg_slice = seg_mask[:, :, i]  # Coronal slice

            # Ensure the segmentation mask is binary
            seg_slice_binary = (seg_slice > 0).astype(np.uint8)

            # Create a mask overlay for the current segmentation
            seg_rgb = np.zeros_like(ct_rgb)
            seg_rgb[seg_slice_binary == 1] = SEG_COLORS.get(seg_name, [1, 1, 1])  # Default to white if not in color map

            # Add the current segmentation to the combined overlay
            combined_seg_rgb = np.clip(combined_seg_rgb + seg_rgb, 0,
                                       1)  # Combine multiple masks and clip to valid range

        # Alpha blend the combined segmentation mask onto the CT slice
        overlay = np.clip(ct_rgb * (1 - alpha) + combined_seg_rgb * alpha, 0, 1)  # Adjust alpha blending and clip

        # Plot the CT image with the combined segmentation mask overlay
        plt.imshow(overlay, cmap='gray', aspect=aspect_ratio)
        plt.title(f'Patient {patient_id} - {plane.capitalize()} Plane - Slice {i + 1}')
        plt.axis('off')

        # Show the plot in Jupyter Notebook
        plt.show()

        # Save the plot
        plt.savefig(f'{output_dir}/{patient_id}_{plane}_slice_{i + 1}.png')
        plt.close()
