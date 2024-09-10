import os
import torch
import torchvision
from PIL import Image
from DICNet3+ import DICNet3
import numpy as np
import matplotlib.pyplot as plt


def load_and_transform_image(image_path, device):
    image = Image.open(image_path).convert('L')
    image = torchvision.transforms.ToTensor()(image)
    return image.to(device)


def visualize_displacement(displacement, title, save_path, factor=0.177):
    plt.figure(figsize=(10, 5))

    # Plot U direction
    plt.subplot(1, 2, 1)
    u_displacement = displacement[0][0].cpu().numpy()
    img1 = plt.imshow(u_displacement, cmap='jet')
    plt.title(f"{title} - U direction")

    # Create colorbar with 5 values and adjust ticks with factor
    cbar1 = plt.colorbar(img1, shrink=0.5, ticks=np.linspace(u_displacement.min(), u_displacement.max(), 5))
    cbar1.ax.set_yticklabels(np.round(cbar1.get_ticks() * factor, 2))
    plt.xticks([]), plt.yticks([])

    # Plot V direction
    plt.subplot(1, 2, 2)
    v_displacement = displacement[0][1].cpu().numpy()
    img2 = plt.imshow(v_displacement, cmap='jet')
    plt.title(f"{title} - V direction")

    # Create colorbar with 5 values and adjust ticks with factor
    cbar2 = plt.colorbar(img2, shrink=0.5, ticks=np.linspace(v_displacement.min(), v_displacement.max(), 5))
    cbar2.ax.set_yticklabels(np.round(cbar2.get_ticks() * factor, 2))
    plt.xticks([]), plt.yticks([])

    # Save the figure
    plt.savefig(save_path)
    plt.show()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model
    net = DICNet3().to(device)
    try:
        net.load_state_dict(torch.load('Z:/YYL/Test1/DICNet3+/model_state_dict_epo500.pth',map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    total_params = sum(p.numel() for p in net.parameters())
    print("Number of parameters: %.2fM" % (total_params / 1e6))

    net.eval()
    x=100
    with torch.no_grad():
        ref_image = load_and_transform_image('Z:/YYL/Test1/DICNet3+/test/image_0075.png',device)
        def_image = load_and_transform_image('Z:/YYL/Test1/DICNet3+/test/image_0100.png', device)
        # Concatenate reference and deformed image
        input_image = torch.cat((ref_image, def_image), dim=0).unsqueeze(dim=0)
        input_image = torchvision.transforms.Normalize(mean=[0.5295, 0.5295], std=[0.4562, 0.4562])(input_image)

        # Predict displacement field
        predicted_displacement = net(input_image)

        # Visualize predicted displacement
        save_dir=f'Z:/YYL/Test1/DICNet3+/test/test{x:05d}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pre_save_path=os.path.join(save_dir,'pre.png')

        visualize_displacement(predicted_displacement, "Predicted Displacement",pre_save_path)
