from itertools import permutations
from math import sqrt
from PIL import Image, ImageDraw, ImageFont
import random
import os


num_photos = 10


# Ensure directories exist
os.makedirs("photos", exist_ok=True)
os.makedirs("photos_unsolved", exist_ok=True)

# Function to calculate distance between two points
def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Solve the TSP using brute force
def solve_tsp(points):
    min_path_length = float('inf')
    best_path = None
    for permutation in permutations(points):
        path_length = sum(distance(permutation[i], permutation[i+1]) for i in range(len(permutation)-1))
        if path_length < min_path_length:
            min_path_length = path_length
            best_path = permutation
    return best_path, min_path_length

for iteration in range(num_photos):
    # Generate 10 random points
    points = [(random.random(), random.random()) for _ in range(10)]

    # Solve the TSP
    best_path, min_path_length = solve_tsp(points)

    # Function to create and save an image
    def create_image(scaled_points, include_path=True, include_text=True, filename="photo.png"):
        img = Image.new('L', (256, 256), 'white')
        draw = ImageDraw.Draw(img)

        if include_path:
            for i in range(len(scaled_points)-1):
                draw.line([scaled_points[i], scaled_points[i+1]], fill=0, width=2)
        
        for point in scaled_points:
            draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=0)

        if include_text:
            font_size = 12
            font = ImageFont.truetype("arial.ttf", font_size)
            text = f"{min_path_length:.2f}"
            text_width, text_height = draw.textsize(text, font=font)
            draw.text((256 - text_width - 10, 256 - text_height - 10), text, fill=0, font=font)
        
        img.save(filename)

    # Scale points to fit in the 256x256 image
    scaled_points = [(int(x*255), int(y*255)) for x, y in best_path]

    # Save the solved image
    solved_filename = f"photos/photo_{iteration:03d}.png"
    create_image(scaled_points, include_path=True, include_text=True, filename=solved_filename)

    # Save the unsolved image
    unsolved_filename = f"photos_unsolved/photo_{iteration:03d}_unsolved.png"
    create_image(scaled_points, include_path=False, include_text=False, filename=unsolved_filename)
