import os
import cv2
import boto3
from zipfile import ZipFile

# Define the AWS S3 bucket name
bucket_name = 'photocrafter'

# Define the AWS S3 client
s3 = boto3.client('s3')

# Define the directory paths
original_dir = 'original'
processed_dir = 'processed'

# Create the processed directory if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

desired_size = 512

counter = 1
for filename in os.listdir(original_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image
        img = cv2.imread(os.path.join(original_dir, filename), cv2.IMREAD_UNCHANGED)
        
        # Get the aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        
        # Calculate the new dimensions
        if aspect_ratio > 1:
            new_width = desired_size
            new_height = int(desired_size / aspect_ratio)
        else:
            new_width = int(desired_size * aspect_ratio)
            new_height = desired_size
        
        # Ensure the new dimensions are not zero
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Resize the image while keeping the aspect ratio constant
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        
        # Calculate the margins to be added
        top = bottom = (desired_size - new_height) // 2
        left = right = (desired_size - new_width) // 2
        
        # Add the margins to the image
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Save the processed image in the processed directory in jpg format with the new name
        cv2.imwrite(os.path.join(processed_dir, 'sks (' + str(counter) + ').jpg'), img)
        counter += 1

# Create a zip file of the processed directory
with ZipFile('processed.zip', 'w') as zipf:
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            zipf.write(os.path.join(root, file))
try:
    s3.head_bucket(Bucket=bucket_name)
except boto3.exceptions.botocore.exceptions.ClientError:
    # The bucket does not exist, create it
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={
            'LocationConstraint': 'ap-northeast-1'
        }
    )

# Generate a unique name for the zip file
import uuid
zip_file_name = f"dataset-{uuid.uuid4()}.zip"

# Upload the zip file to AWS S3
with open('processed.zip', 'rb') as data:
    s3.upload_fileobj(data, bucket_name, zip_file_name)

# Generate a presigned URL for the uploaded zip file
url = s3.generate_presigned_url(
     ClientMethod='get_object',
     Params={
         'Bucket': bucket_name,
         'Key': zip_file_name
     }
 )
print(f"Publicly accessible URL: {url}")

# Add the DATASET_URL to the .env file
with open('.env', 'r+') as f:
    lines = f.readlines()
    f.seek(0)
    for line in lines:
        if "DATASET_URL" not in line:
            f.write(line)
    f.write(f"\nDATASET_URL={url}")
    f.truncate()
