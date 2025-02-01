import os
import sys
import numpy as np
import open3d as o3d
import cv2
import subprocess
import json
from pathlib import Path
import shutil

class BetterModelCreator:
    def __init__(self):
        self.input_folder = "photogrammetry_images"
        self.output_folder = "3d_model_output"
        
        # Create folders if they don't exist
        for folder in [self.input_folder, self.output_folder]:
            os.makedirs(folder, exist_ok=True)

    def create_3d_model(self):
        print("Starting 3D reconstruction using Structure from Motion...")
        
        # Load images
        image_paths = list(Path(self.input_folder).glob("*.jpg"))
        if len(image_paths) < 10:
            print(f"Error: Found only {len(image_paths)} images. Need at least 10 images.")
            return False

        try:
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            
            # Initialize feature matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Load and process images
            print("\nProcessing images...")
            images = []
            keypoints_list = []
            descriptors_list = []
            
            for img_path in image_paths:
                # Load and convert to grayscale
                img = cv2.imread(str(img_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find keypoints and descriptors
                kp, des = sift.detectAndCompute(gray, None)
                
                images.append(img)
                keypoints_list.append(kp)
                descriptors_list.append(des)
                print(f"Processed {img_path.name} - Found {len(kp)} keypoints")

            # Match features between consecutive images
            print("\nMatching features between images...")
            points_3d = []
            colors_3d = []
            
            for i in range(len(images)-1):
                matches = flann.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                print(f"Found {len(good_matches)} good matches between images {i} and {i+1}")
                
                if len(good_matches) > 10:
                    # Get matched keypoints
                    src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in good_matches])
                    
                    # Find essential matrix
                    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.),
                                                  method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    
                    # Recover relative camera pose
                    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
                    
                    # Triangulate points
                    P1 = np.array([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0]])
                    P2 = np.hstack((R, t))
                    
                    points_4d = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
                    points_3d_new = (points_4d[:3, :] / points_4d[3, :]).T
                    
                    # Get colors from first image
                    for j, match in enumerate(good_matches):
                        if mask[j]:
                            x, y = map(int, src_pts[j])
                            if 0 <= x < images[i].shape[1] and 0 <= y < images[i].shape[0]:
                                color = images[i][y, x] / 255.0
                                points_3d.append(points_3d_new[j])
                                colors_3d.append(color)

            # Convert lists to numpy arrays
            points_3d = np.array(points_3d, dtype=np.float64)
            colors_3d = np.array(colors_3d, dtype=np.float64)
            
            print(f"\nReconstructed {len(points_3d)} 3D points")
            
            if len(points_3d) < 100:
                print("Error: Not enough points for reconstruction")
                return False

            # Create point cloud
            print("\nCreating point cloud...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors_3d)
            
            # Process point cloud
            print("Processing point cloud...")
            print(f"Initial size: {len(pcd.points)} points")
            
            # Remove outliers
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            print(f"After downsampling: {len(pcd.points)} points")
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Create mesh
            print("\nCreating mesh...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # Clean up mesh
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.compute_vertex_normals()
            
            # Save results
            output_path = os.path.join(self.output_folder, "model.obj")
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"\nModel saved to: {output_path}")
            
            # Visualize
            print("\nOpening visualization...")
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            
            # Improve visualization settings
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True
            opt.background_color = np.asarray([0.2, 0.2, 0.2])
            
            print("\nControls:")
            print("- Left mouse: Rotate")
            print("- Right mouse: Pan")
            print("- Mouse wheel: Zoom")
            print("- Q: Close visualization")
            
            vis.run()
            vis.destroy_window()
            
            return True

        except Exception as e:
            print(f"Error during reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    creator = BetterModelCreator()
    creator.create_3d_model()

# Create necessary folders
folders = ["photogrammetry_images", "3d_model_output"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}") 