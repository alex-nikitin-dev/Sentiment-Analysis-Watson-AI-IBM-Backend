"""
This module provides a function to analyze sentiment of a given text
using the Watson Sentiment Analysis API.
"""

import requests
from typing import Dict, Optional


def sentiment_analyzer(text_to_analyse: str) -> Dict[str, Optional[str]]:
    """
    Analyzes the sentiment of the provided text
    using the Watson Sentiment Analysis API.

    Parameters:
        text_to_analyse (str): The text to analyze.

    Returns:
        Dict[str, Optional[str]]: A dictionary
        containing the sentiment 'label' and 'score'.
    """
    # Define the URL for the sentiment analysis API
    url = (
        'https://sn-watson-sentiment-bert.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'
    )
    # Create the payload with the text to be analyzed
    myobj = {"raw_document": {"text": text_to_analyse}}
    # Set the headers with the required model ID for the API
    header = {"grpc-metadata-mm-model-id":
              "sentiment_aggregated-bert-workflow_lang_multi_stock"}
    # Initialize label and score to None
    label = None
    score = None

    try:
        # Make a POST request to the API with the payload and headers
        response = requests.post(url, json=myobj, headers=header, timeout=10)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response from the API
            formatted_response = response.json()
            # Extract the label and score from the response
            label = formatted_response['documentSentiment']['label']
            score = formatted_response['documentSentiment']['score']
        else:
            # Handle non-200 responses
            print((f"Error: Received status code {response.status_code}."
                   f" Response: {response.text}"))
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Return the label and score in a dictionary
    return {'label': label, 'score': score}
