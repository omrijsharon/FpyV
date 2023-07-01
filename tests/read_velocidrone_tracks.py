import base64

def read_decode_file(file_path):
    with open(file_path, 'r') as file:
        encoded_data = file.read()
    decoded_data = base64.b64decode(encoded_data)
    return decoded_data

def main():
    file_path = r'C:\Users\omri_\OneDrive\Documents\Kfar Hes.trk'
    decoded_data = read_decode_file(file_path)
    print(decoded_data)
    # Continue with parsing the decoded_data
    # ...

if __name__ == "__main__":
    main()