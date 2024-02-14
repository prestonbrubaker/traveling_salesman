import math
import numpy as np
import pillow
import itertools

# Step 1: Generate 10 random points
np.random.seed(0)  # For reproducibility
points = np.random.rand(10, 2)

# Step 2: Solve the TSP using a brute-force approach
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(points):
    return sum(distance(point, points[index + 1]) for index, point in enumerate(points[:-1]))

min_path = None
min_distance = float('inf')
for permutation in itertools.permutations(points):
    current_distance = total_distance(permutation)
    if current_distance < min_distance:
        min_distance = current_distance
        min_path = permutation

# Step 3: Draw the points and path
img = Image.new('L', (256, 256), 'white')
draw = ImageDraw.Draw(img)

# Scale points to image size
scaled_points = [(x * 255, y * 255) for x, y in min_path]

# Draw path
for i in range(len(scaled_points) - 1):
    draw.line([scaled_points[i], scaled_points[i + 1]], fill=128, width=2)

# Draw points
for x, y in scaled_points:
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=0)

# Add total length text
font = ImageFont.load_default()  # Using default font
text = f"{min_distance:.2f}"
text_width, text_height = draw.textsize(text, font=font)
draw.text((256 - text_width - 10, 256 - text_height - 10), text, fill=128, font=font)

# Step 4: Save the image
img_path = "photos"
img.save(img_path)

img_path, min_distance
