from PIL import Image, ImageDraw, ImageFont
import argparse

def genimage(text, filename, textcolor, bgcolor):

    font_size = 40                    # Font size
    text_color = textcolor     # White text color (RGB format)
    background_color = bgcolor  # Blue background (RGB format)

    # Set up font (ensure font path and font size are correct)
    try:
        # Load a TrueType font (you can replace this with the path to any font file on your system)
        print("font setup")
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)  # For Linux
        # On Windows, you could use: font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
    except IOError:
        print("font not found")
        font = ImageFont.load_default()  # Default font if custom one is not found

    # Create a dummy image to calculate the text size
    dummy_img = Image.new('RGB', (1, 1))  # Creating a dummy image just for size calculation
    draw = ImageDraw.Draw(dummy_img)

    # Calculate the bounding box of the text (left, top, right, bottom)
    text_bbox = draw.textbbox((0, 0), text, font=font)

    # Get the width and height from the bounding box
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Image properties: adjust width based on text width
    image_width = text_width + 40  # Adding some padding (20 pixels on each side)
    image_height = (text_height + 40 )  # Adding padding for height

    # Create a new image with the adjusted width and height
    img = Image.new('RGB', (image_width, image_height), background_color)

    # Initialize ImageDraw object to draw on the image
    draw = ImageDraw.Draw(img)

    # Calculate the position to center the text
    text_x = (image_width - text_width) / 2
    text_y = 0.3 *(image_height - text_height)

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Save the generated image
    img.save(filename + '.png')

    # Show the image (optional, to preview it)
    # img.show()

def combine_images_horizontally(images):
    # Open all images
    opened_images = [Image.open(image) for image in images]
    
    # Get the total width and the maximum height
    total_width = sum(image.width for image in opened_images)
    max_height = max(image.height for image in opened_images)
    
    # Create a new image with the total width and maximum height
    combined_image = Image.new("RGB", (total_width, max_height))
    
    # Paste each image into the combined image
    x_offset = 0
    for image in opened_images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width
    
    return combined_image


def readmestatus(PR_num, ISSUE_num, RELEASE_num):
    genimage("Pull Requests", "./pics/PR",(255, 255, 255), (0, 0, 0))
    genimage(PR_num, "./pics/PR_num",(0, 0, 0), (164, 193, 162))
    genimage("Version", "./pics/Version",(255, 255, 255), (0, 0, 0))
    genimage(RELEASE_num, "./pics/Version_num",(0, 0, 0), (197, 226, 162))
    genimage("Issues", "./pics/Issues",(255, 255, 255), (0, 0, 0))
    genimage(ISSUE_num, "./pics/Issue_num",(255, 255, 255), (56, 66, 132))

    combined_image = combine_images_horizontally(["./pics/PR.png", 
                                "./pics/PR_num.png",
                                "./pics/Version.png", 
                                "./pics/Version_num.png", 
                                "./pics/Issues.png", 
                                "./pics/Issue_num.png"])

    # Save or display the combined image
    combined_image.save("./pics/Readme_status.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process readme status.")
    parser.add_argument("pr_status", type=str, help="The PR status argument (e.g., '0 Open')")
    parser.add_argument("issue_status", type=str, help="The issue count argument (e.g., '5 Issues')")
    parser.add_argument("release_status", type=str, help="The release number argument (e.g., '1.0.0')")
    args = parser.parse_args()

    readmestatus(args.pr_status, args.issue_status, args.release_status)