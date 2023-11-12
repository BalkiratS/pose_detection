import argparse as ap
import csv
import os
from typing import Any
import cv2
import mediapipe.python.solutions.pose as pose
import sys

# Description of the program
PROG_DESCRIPTION = "Program takes all the image files from the given directory and runs the BlazePose pose detection model on the images, the results from each image are collaboratively stored in a csv file with the same name as the directory."

# Landmarks to be detected by BlazePose
LANDMARKS = [
    pose.PoseLandmark.NOSE,
    pose.PoseLandmark.LEFT_SHOULDER,
    pose.PoseLandmark.RIGHT_SHOULDER,
    pose.PoseLandmark.LEFT_ELBOW,
    pose.PoseLandmark.RIGHT_ELBOW,
    pose.PoseLandmark.LEFT_WRIST,
    pose.PoseLandmark.RIGHT_WRIST,
    pose.PoseLandmark.LEFT_HIP,
    pose.PoseLandmark.RIGHT_HIP,
    pose.PoseLandmark.LEFT_KNEE,
    pose.PoseLandmark.RIGHT_KNEE,
    pose.PoseLandmark.LEFT_ANKLE,
    pose.PoseLandmark.RIGHT_ANKLE,
]

# Function to parse command line arguments
def parse_args():
    """Parses the program positional and optional arguments."""
    
    parser = ap.ArgumentParser(prog="pose_detection", description=PROG_DESCRIPTION,)
    parser.add_argument("directory")

    parser.add_argument(
        "-o", "--output", metavar="FILENAME", help="name of the csv file to output"
    )

    parser.add_argument(
        "-p", "--pixels", action="store_true", help="store the data in pixels",
    )

    parser.add_argument(
        "-l", "--logs", action="store_true", help="print debug logs as it's processing"
    )

    parser.add_argument(
        "-t",
        "--test-data",
        action="store_true",
        help="select test directory instead of train",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="shows the landmark on output images",
    )

    args = parser.parse_args()

    return args

# Function to create a debug directory for storing debug images
def create_debug_directory():
    """Creates the debug directory if it does not exist."""
    if os.path.exists("debug"):
        os.rmdir("debug")

    return os.mkdir("debug")

# Function to load the BlazePose model
def get_model():
    """Loads the blaze pose model."""
    model = pose.Pose(
        static_image_mode=True, model_complexity=2, enable_segmentation=True,
    )

    return model

# Function to determine the directory based on test or train data
def get_directory(args: ap.Namespace):
    directory = args.directory

    if args.test_data:
        directory += "/test/"
    else:
        directory += "/train/"

    return directory

# Function to get the list of image filenames in a directory
def get_image_filenames(directory: str):
    """Lists the image files inside the directory."""
    return os.listdir(directory)

# Function to create the header for the CSV file
def get_csv_header():
    """Creates the header for the csv file."""
    header: list[str] = ["filename"]

    for landmark in LANDMARKS:
        landmark_key = landmark.name.lower()
        header.extend(
            [f"{landmark_key}_x", f"{landmark_key}_y", f"{landmark_key}_score"]
        )

    header.extend(["class_no", "class_name"])

    return header

# Function to determine the output CSV filename
def get_csv_filename(args: ap.Namespace) -> str:
    """Gets the filename for the output csv."""
    # If the filename is provided by the user, use that.
    if args.output:
        return args.output

    # Else use the directory's name by default.
    default = f"{args.directory}.csv"
    return default

# Function to get the landmark positions from a given image file
def get_positions(model: pose.Pose, filename: str, args: ap.Namespace):
    """Reads the file with the given name, and returns the positional data from the model."""
    # Read the image from the file.
    image = cv2.imread(filename)

    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Process the image from the model.
    results = model.process(image_array)

    # If not landmarks present, return from function.
    if results.pose_landmarks == None:
        print(f"Unable to read pose landmarks from file: {filename}", file=sys.stderr)
        return

    # Output data resembles how the csv row looks like, it includes the filename, and each
    # landmark's x, y position and it's score.
    data = {"filename": filename}

    # Create a debug copy of the image.
    debug_image = image.copy()

    # For all the landmarks present in blaze pose.
    for landmark in LANDMARKS:
        result_landmark = results.pose_landmarks.landmark[landmark]

        if args.pixels:
            landmark_x = result_landmark.x * height
            landmark_y = result_landmark.y * width
        else:
            landmark_x = result_landmark.x
            landmark_y = result_landmark.y

        landmark_key = landmark.name.lower()

        # Add each landmark data to the dictionary.
        data[f"{landmark_key}_x"] = landmark_x
        data[f"{landmark_key}_y"] = landmark_y
        data[f"{landmark_key}_score"] = result_landmark.visibility

        # If the debug mode is selected, draw the circle points identifying the joints.
        if args.debug:
            cv2.circle(
                debug_image,
                (int(result_landmark.x * width), int(result_landmark.y * height)),
                8,
                (0, 0, 255),
                -1,
            )

    # If debug mode is selected, write the image with the joints drawn into a file.
    if args.debug:
        file = f"{os.getcwd()}/debug/{filename.replace('/', '_')}"
        cv2.imwrite(file, debug_image)

    return data

# Function to record image data from the provided model
def record_image_data(args: ap.Namespace):
    """Records the image data from the provided model."""
    model = get_model()

    directory = get_directory(args)
    good_filenames = get_image_filenames(f"{directory}/good")
    bad_filenames = get_image_filenames(f"{directory}/bad")

    csv_filename = get_csv_filename(args)
    csv_header = get_csv_header()

    # Open the csv file.
    csv_file = open(csv_filename, "w")
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)

    # Write the csv header.
    writer.writeheader()

    # Locate positions and record the data into csv.
    for filename in good_filenames:
        if ".DS_Store" in filename:
            continue

        positions: Any = get_positions(model, f"{directory}/good/{filename}", args)
        positions["class_no"] = 1
        positions["class_name"] = "good"

        writer.writerow(positions)

        if args.logs:
            print(f"Successfully finished {filename}")

    for filename in bad_filenames:
        if ".DS_Store" in filename:
            continue

        positions: Any = get_positions(model, f"{directory}/bad/{filename}", args)
        positions["class_no"] = 0
        positions["class_name"] = "bad"

        writer.writerow(positions)

        if args.logs:
            print(f"Successfully finished {filename}")

    csv_file.close()
    model.close()

# Main entry point of the program
def main():
    """Main entrypoint."""
    args = parse_args()

    if args.debug:
        create_debug_directory()

    record_image_data(args)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
