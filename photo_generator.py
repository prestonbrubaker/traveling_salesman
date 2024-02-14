from itertools import permutations
from math import sqrt
from PIL import Image, ImageDraw, ImageFont
import random

# Generate 10 random points
points = [(random.random(), random.random()) for _ in range(10)]

# Function to calculate distance between two points
def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Solve the TSP using brute force - find the shortest path through all points
def solve_tsp(points):
    min_path_length = float('inf')
    best_path = None
    for permutation in permutations(points):
        path_length = sum(distance(permutation[i], permutation[i+1]) for i in range(len(permutation)-1))
        if path_length < min_path_length:
            min_path_length = path_length
            best_path = permutation
    return best_path, min_path_length

# Solve the TSP
best_path, min_path_length = solve_tsp(points)

# Create a 256x256 PNG image in greyscale
img = Image.new('L', (256, 256), 'white')
draw = ImageDraw.Draw(img)

# Scale points to fit in the 256x256 image
scaled_points = [(int(x*255), int(y*255)) for x, y in best_path]

# Draw the path
for i in range(len(scaled_points)-1):
    draw.line([scaled_points[i], scaled_points[i+1]], fill=0, width=2)

# Draw the points as circles
for point in scaled_points:
    draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=0)

# Draw the total length of the path in the bottom right corner
font_size = 12
font = ImageFont.truetype("arial.ttf", font_size)
text = f"{min_path_length:.2f}"
text_width, text_height = draw.textsize(text, font=font)
draw.text((256 - text_width - 10, 256 - text_height - 10), text, fill=0, font=font)

# Save the image
img_path = "photos/photo.png"
img.save(img_path)

img_path
