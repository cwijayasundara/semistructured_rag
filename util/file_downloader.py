import requests

# URL of the web page
url = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2025"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Get the HTML content
    html_content = response.text

    # Specify the file name
    file_name = "../docs/nvidia_financial_results_q1_fiscal_2025.html"

    # Write the HTML content to a file
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"HTML content has been downloaded and saved as {file_name}")
else:
    print(f"Failed to retrieve the web page. Status code: {response.status_code}")