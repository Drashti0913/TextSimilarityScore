# Text Similarity Model

This repository contains a text similarity scoring model designed to assess the semantic similarity between two paragraphs. The model leverages the TF-IDF technique combined with cosine similarity to provide a nuanced comparison of textual content.

## Project Structure

- `text_similarity_model.py`: Contains the core logic for the model, including data preprocessing, TF-IDF vectorization, and similarity calculation.
- `app.py`: Flask application for the RESTful API to interact with the text similarity model.
- `requirements.txt`: List of dependencies required to run the application.
- `Procfile`: Specifies the commands that are executed by the app on startup on Heroku (optional if deploying on AWS EC2).

## Installation

To set up this project, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip
- AWS EC2 instance (for deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-similarity-model.git
   cd text-similarity-model
2. Install the required packages:
    pip install -r requirements.txt

3. Run the Flask application locally:
   python app.py


### Deployment on AWS EC2
To deploy the application on an AWS EC2 instance, follow the AWS documentation to set up an EC2 instance, configure the security groups, and deploy the Flask application.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or find a bug.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Contact
If you have any questions, please open an issue or contact drashtibhavsar09@gmail.com.


---

Feel free to adjust the content as per your project specifics and GitHub settings, such as repository links and deployment details. If there are specific sections or details you want to be added or modified, just let me know!
