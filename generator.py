import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# Model Parameters
latent_dim = 256  # Example latent space dimension
variability_scale = 5.0  # Adjust this value to increase or decrease variability of the generated images



class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # Output: 16x128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 256x16x16
            nn.ReLU(),
            nn.Flatten(),  # Flatten for linear layer input
            nn.Linear(128*16*16, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),  # Unflatten to 256x16x16 for conv transpose input
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Output: 1x256x256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var




# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device))


def load_model(path, latent_dim, device):
    """Load the trained VAE model from a given path."""
    model = VariationalAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def reparameterize(mu, log_var, variability_scale=1.0):
    """Reparameterize with an additional scale for variability."""
    std = torch.exp(0.5 * log_var) * variability_scale
    eps = torch.randn_like(std)
    return mu + eps * std

def generate_images_with_random_latents(model, num_images, folder_path, latent_dim, variability_scale=1.0):
    """Generate and save images using randomly sampled latent variables."""
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            # Sample random latent variables
            random_latent_vector = torch.randn(1, latent_dim, device=device) * variability_scale
            generated_image = model.decode(random_latent_vector)

            # Convert the generated image to PIL format and save
            generated_image = generated_image.squeeze(0).cpu()  # Ensure it's a single image tensor
            img = transforms.ToPILImage()(generated_image)
            img.save(os.path.join(folder_path, f"generated_image_{i+1}.png"))

if __name__ == "__main__":
    # Load the trained model
    model_path = 'variational_autoencoder.pth'
    vae_model = load_model(model_path, latent_dim, device)

    # Generate images using random latents
    num_generated_images = 500
    output_folder = 'generated_photos'
    

    generate_images_with_random_latents(vae_model, num_generated_images, output_folder, latent_dim, variability_scale)

    print(f"Generated {num_generated_images} images in '{output_folder}' folder with variability scale {variability_scale}.")
