from PIL import Image
import os

def check_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.convert('RGB')
            return True, None
    except Exception as e:
        return False, str(e)

for file_name in os.listdir('./data/Negative'):
    file_path = os.path.join('./data/Negative', file_name)
    success, error_message = check_image(file_path)
    if not success:
        print(f'{file_name} is corrupted: {error_message}')
