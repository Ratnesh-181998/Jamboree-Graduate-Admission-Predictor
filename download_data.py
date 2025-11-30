import requests

url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/839/original/Jamboree_Admission.csv"
output_file = "Jamboree_Admission.csv"   # Corrected filename to match the actual data

response = requests.get(url)

with open(output_file, "wb") as f:
    f.write(response.content)

print("Download completed! File saved as", output_file)
