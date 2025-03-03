import pandas as pd
import os
from typing import Tuple, Optional
from google import generativeai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure APIs
client = OpenAI()
generativeai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration settings
CONFIG = {
    'openai_model': 'gpt-4o-mini',
    'gemini_model': 'gemini-2.0-flash-thinking-exp',
    'batch_size': 10,
    'test_rows': 20,
    'use_openai': False  # Set to True to use OpenAI instead of Gemini
}

# Define possible services and activities
SERVICES = [
    "Evernote", "Asana", "Vercel", "Netlify", "Google Cloud",
    "Microsoft", "Unknown Service"
]

ACTIVITIES = [
    "Authentication",
    "Login",
    "Logout",
    "Deployment",
    "Build",
    "Access",
    "Upload",
    "Download",
    "Email Sending",
    "Email Receiving",
    "Attachment Upload",
    "Attachment Download",
    "Message",
    "Call",
    "Meeting",
    "Guide Viewing",
    "Guide Completion",
    "API Request",
    "Data Sync",
    "Configuration Update",
    "Health Check",
    "Unknown Activity"
]


def create_prompt(row: pd.Series) -> str:
    """Create a prompt for the LLM based on the row data."""
    return f"""Based on the following HTTP request data, classify the service being accessed and the activity being performed.

    Host: {row['headers_Host']}
    Method: {row['method']}
    URL: {row['url']}
    Content-Type: {row['requestHeaders_Content_Type']}
    Accept: {row['requestHeaders_Accept']}
    Origin: {row.get('requestHeaders_Origin', 'N/A')}
    Referer: {row.get('requestHeaders_Referer', 'N/A')}

    Consider these aspects for classification:
    - Authentication related URLs contain: auth, signin, login, sso, token
    - Upload activities use PUT or POST methods
    - Download activities typically use GET method
    - Message activities contain: message, chat
    - Meeting activities contain: meeting, schedule

    Available Services: {', '.join(SERVICES)}
    Available Activities: {', '.join(ACTIVITIES)}

    Respond in the following format only:
    Service: <service_name>
    Activity: <activity_name>
    """


def get_openai_classification(prompt: str) -> Tuple[str, str]:
    """Get classification using OpenAI API."""
    try:
        completion = client.chat.completions.create(
            model=CONFIG['openai_model'],
            messages=[
                {"role": "system",
                 "content": "You are a classifier that categorizes HTTP requests into services and activities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = completion.choices[0].message.content

        # Parse the response
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()

        return service, activity
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Unknown Service", "Unknown Activity"


def get_gemini_classification(prompt: str) -> Tuple[str, str]:
    """Get classification using Gemini API."""
    try:
        model = generativeai.GenerativeModel(CONFIG['gemini_model'])
        response = model.generate_content(prompt)
        result = response.text

        # Parse the response
        service = result.split("Service:")[1].split("Activity:")[0].strip()
        activity = result.split("Activity:")[1].strip()

        return service, activity
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Unknown Service", "Unknown Activity"


def label_dataset(csv_path: str, use_openai: bool = True) -> pd.DataFrame:
    """
    Label the dataset using the specified LLM API.

    Args:
        csv_path: Path to the input CSV file
        use_openai: If True, use OpenAI API; if False, use Gemini API
    """
    # Read the dataset
    df = pd.read_csv(csv_path)

    # Initialize new columns if they don't exist
    if 'predicted_service' not in df.columns:
        df['predicted_service'] = None
    if 'predicted_activity' not in df.columns:
        df['predicted_activity'] = None

    # Get classification function based on selected API
    classify_func = get_openai_classification if use_openai else get_gemini_classification

    # Process rows that haven't been labeled yet
    for idx, row in df.iterrows():
        if pd.isna(row['predicted_service']) or pd.isna(row['predicted_activity']):
            prompt = create_prompt(row)
            service, activity = classify_func(prompt)

            df.at[idx, 'predicted_service'] = service
            df.at[idx, 'predicted_activity'] = activity

            print(f"Processed row {idx}: Service={service}, Activity={activity}")

            # Save progress after each batch
            if idx % CONFIG['batch_size'] == 0:
                df.to_csv(csv_path, index=False)
                print(f"Progress saved at row {idx}")

    # Save final results
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    # Example usage
    csv_path = "data/evernote_test.csv"

    # Read only first n rows for testing
    df = pd.read_csv(csv_path).head(CONFIG['test_rows'])
    test_csv_path = "data/evernote_test_sample.csv"
    df.to_csv(test_csv_path, index=False)

    print(f"Starting classification using {'OpenAI' if CONFIG['use_openai'] else 'Gemini'} API...")
    print(f"Testing with first {CONFIG['test_rows']} rows...")

    df = label_dataset(test_csv_path, use_openai=CONFIG['use_openai'])
    print("Classification completed!")

    # Print results summary
    print("\nResults Summary:")
    print("Services found:", df['predicted_service'].value_counts())
    print("\nActivities found:", df['predicted_activity'].value_counts())
