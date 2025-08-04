import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import re
import os
import joblib

# --- IMPORTANT CHANGE FOR RENDER ---
# Use a relative path for data files. Create a 'DataFiles' folder
# at the root of your project directory, next to app.py.
# All CSVs should be inside this 'DataFiles' folder.
base_dir = os.path.join(os.path.dirname(__file__), "DataFiles")

LEGIT_URLS_FILE = os.path.join(base_dir, "legitimateurls.csv")
PHISH_URLS_FILE = os.path.join(base_dir, "phishurls.csv")
PHISH_URLS_FILE_ADDITIONAL = os.path.join(base_dir, "phishurls1.csv")
PHISHING_DATA_FILE = os.path.join(base_dir, "phishing.csv")

# The model file should be in the root directory for easy access by Render
MODEL_FILE = 'phishing_model.pkl'

app = Flask(__name__)

# --- Feature Extraction Function ---
def extract_features_from_url(url):
    """
    Extracts a set of features from a given URL, designed to be consistent
    with the feature set in the phishing.csv file. This is crucial for
    unified model training and prediction.
    It now normalizes 'www.' for subdomain calculation and removes the HTTPS feature.
    
    Args:
        url (str): The URL to analyze.
        
    Returns:
        pd.Series: A pandas Series with extracted features.
    """
    if not isinstance(url, str):
        url = ''

    processed_url = url
    # Ensure a protocol is present for urlparse to work correctly
    if not re.match(r"^[a-zA-Z]+://", url):
        processed_url = "http://" + url

    parsed_url = urlparse(processed_url)
    hostname = parsed_url.hostname or ''
    
    # Handle cases where urlparse might not extract hostname correctly for some malformed inputs
    if not hostname and '.' in url:
        match = re.match(r"^(?:https?://)?([^/]+)", url)
        if match:
            hostname = match.group(1)

    # Normalize hostname by stripping 'www.' before subdomain calculation
    # This makes 'www.example.com' and 'example.com' have the same subdomain count
    normalized_hostname = hostname.replace('www.', '') if hostname.startswith('www.') else hostname

    # Calculate subdomains
    if not normalized_hostname:
      subdomains = 0
    else:
      parts = [s for s in normalized_hostname.split('.') if s]
      if len(parts) > 2: # e.g., sub.domain.com -> 1 subdomain
          subdomains = len(parts) - 2 
      elif len(parts) == 2: # e.g., domain.com -> 0 subdomains
          subdomains = 0
      else: # e.g., localhost or single word domain
          subdomains = 0

    features = {
        'UsingIP': 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname) else -1,
        'LongURL': 1 if len(url) > 75 else -1,
        'ShortURL': 1 if len(url) < 30 else -1,
        'Symbol@': 1 if '@' in url else -1,
        'Redirecting//': 1 if url.count('//') > 1 else -1,
        'PrefixSuffix-': 1 if '-' in hostname else -1,
        'SubDomains': subdomains, # Now based on normalized hostname
        # Removed 'HTTPS' feature due to unreliable derivation from input string
    }
    return pd.Series(features)


# --- Model Training/Loading Function ---
def train_model():
    """
    Loads data from all three provided CSVs, extracts features, and trains a
    Random Forest Classifier model on a unified dataset.
    This function will now check if a pre-trained model exists to speed up loading.
    On Render, the model is expected to be pre-trained and shipped with the deployment.
    """
    try:
        # First, try to load the model. This is the primary path for Render.
        if os.path.exists(MODEL_FILE):
            print("Loading pre-trained model...")
            return joblib.load(MODEL_FILE)

        # This block only runs if MODEL_FILE is NOT found (e.g., during local development or initial build)
        print("Model file not found. Attempting to train a new model locally...")
        print("This may take a moment. Ensure all CSV files are in the 'DataFiles' subdirectory.")

        consistent_features = [
            'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
            'PrefixSuffix-', 'SubDomains'
        ]

        phishing_data_df = pd.read_csv(PHISHING_DATA_FILE, encoding='latin1')
        phishing_data_df = phishing_data_df.drop(columns=['Index', 'StatsReport'], errors='ignore') 
        phishing_data_df = phishing_data_df.rename(columns={'class': 'label'})
        phishing_data_df = phishing_data_df[consistent_features + ['label']]

        legit_df = pd.read_csv(LEGIT_URLS_FILE, header=None, names=['url'], encoding='latin1')
        legit_features = legit_df['url'].apply(lambda u: extract_features_from_url(u))
        legit_features['label'] = 1  # 1 for legitimate

        phish_df_main = pd.read_csv(PHISH_URLS_FILE, header=None, names=['url'], encoding='latin1')
        phish_features_main = phish_df_main['url'].apply(lambda u: extract_features_from_url(u))
        phish_features_main['label'] = -1  # -1 for phishing

        phish_features_additional = pd.DataFrame()
        if os.path.exists(PHISH_URLS_FILE_ADDITIONAL):
            print(f"Loading additional phishing URLs from {PHISH_URLS_FILE_ADDITIONAL}...")
            phish_df_additional = pd.read_csv(PHISH_URLS_FILE_ADDITIONAL, header=None, names=['url'], encoding='latin1')
            phish_features_additional = phish_df_additional['url'].apply(lambda u: extract_features_from_url(u))
            phish_features_additional['label'] = -1
        else:
            print(f"Warning: Additional phishing URL file {PHISH_URLS_FILE_ADDITIONAL} not found. Skipping.")

        phish_features_main = phish_features_main[consistent_features + ['label']]
        if not phish_features_additional.empty:
            phish_features_additional = phish_features_additional[consistent_features + ['label']]
            phish_features = pd.concat([phish_features_main, phish_features_additional], ignore_index=True)
        else:
            phish_features = phish_features_main

        legit_features = legit_features[consistent_features + ['label']]
        
        combined_df = pd.concat([phishing_data_df, legit_features, phish_features], ignore_index=True)

        X = combined_df.drop(columns=['label'])
        y = combined_df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test set: {accuracy:.2f}")
        
        joblib.dump(model, MODEL_FILE)
        print("Model trained and saved to file.")

        return model
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Please ensure CSV files are in the 'DataFiles' subdirectory: {e}")
        # On Render, if the model file is not found (and not pre-trained),
        # this means the deployment is missing critical data.
        print("Ensure 'phishing_model.pkl' is pre-trained and included in your deployment, and CSVs are in the 'DataFiles' folder.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model training/loading: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging on Vercel logs
        return None

# Train or load the model once when the application starts
model = train_model()
if model is None:
    print("Failed to load or train model. Application will not run correctly.")

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main HTML page for the phishing checker."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive a URL and return a prediction with probability.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded or trained. Check server logs.'}), 500

    try:
        data = request.get_json()
        url = data['url']
        
        features = extract_features_from_url(url)
        
        # Ensure the feature names and order match what the model expects
        # model.feature_names_in_ will hold the names from when the model was trained.
        # This ensures robustness even if feature extraction order changes slightly.
        feature_values = features.loc[model.feature_names_in_].values.reshape(1, -1)
        
        prediction = model.predict(feature_values)
        probabilities = model.predict_proba(feature_values)

        predicted_class_index = np.where(model.classes_ == prediction[0])[0][0]
        confidence = probabilities[0][predicted_class_index] * 100

        if prediction[0] == 1:
            result = "Legitimate"
            is_phishing = False
        else:
            result = "Phishing"
            is_phishing = True
            
        return jsonify({
            'url': url,
            'prediction': result,
            'is_phishing': is_phishing,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging on Vercel logs
        return jsonify({'error': str(e)}), 500


# Fixed the SyntaxWarning by using a Python raw string literal for the HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=0.8, shrink-to-fit=no">
    <title>Phishing URL Checker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .result-box-transition {
            transition: all 0.5s ease-out;
            opacity: 0;
            transform: translateY(20px);
        }
        .result-box-transition.active {
            opacity: 1;
            transform: translateY(0);
        }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .hover-lift:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        /* Keyframes for the background blob animation */
        @keyframes blob {
            0% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -50px) scale(1.1); }
            66% { transform: translate(-20px, 20px) scale(0.9); }
            100% { transform: translate(0, 0) scale(1); }
        }

        .animate-blob {
            animation: blob 7s infinite cubic-bezier(0.68, -0.55, 0.27, 1.55);
        }

        .animation-delay-2000 { animation-delay: 2s; }
        .animation-delay-4000 { animation-delay: 4s; }

        /* Animation for shaking when phishing */
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        .animate-shake {
            animation: shake 0.6s cubic-bezier(.36,.07,.19,.97) both;
        }

        /* Animation for icon pop-in */
        @keyframes popIn {
            0% { transform: scale(0); opacity: 0; }
            80% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        .icon-pop-in {
            animation: popIn 0.4s ease-out forwards;
        }

        /* Animation for small header icons */
        @keyframes fadeInScale {
            0% { opacity: 0; transform: scale(0.8); }
            100% { opacity: 1; transform: scale(1); }
        }
        .animate-header-icon {
            animation: fadeInScale 0.5s ease-out forwards;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4 gradient-bg relative overflow-hidden">

    <div class="absolute top-0 left-0 w-72 h-72 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob"></div>
    <div class="absolute top-0 right-0 w-72 h-72 bg-yellow-400 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob animation-delay-2000"></div>
    <div class="absolute bottom-0 left-1/4 w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob animation-delay-4000"></div>

    <div class="bg-white rounded-3xl shadow-2xl p-8 max-w-3xl w-full z-10 border border-gray-200 hover-lift">
        <div class="flex flex-col items-center justify-center mb-6 text-indigo-700">
            <svg class="h-16 w-16 mb-2 text-indigo-600 animate-header-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.01 12.01 0 003 21h18a12.01 12.01 0 00-.382-14.016z" />
            </svg>
            <h1 class="text-4xl font-extrabold text-gray-800 tracking-tight flex items-center">
                <svg class="h-8 w-8 mr-2 text-indigo-500 animate-header-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.01 12.01 0 003 21h18a12.01 12.01 0 00-.382-14.016z" />
                </svg>
                Phishing Detector
                <svg class="h-8 w-8 ml-2 text-indigo-500 animate-header-icon animation-delay-2000" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm0-12h.01M12 7l-6 6h12l-6-6z" />
                </svg>
            </h1>
            <p class="text-indigo-500 text-sm mt-1 flex items-center">
                Your Digital Security Companion
                <svg class="h-4 w-4 ml-1 text-indigo-400 animate-header-icon animation-delay-4000" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 20v-3m0 0l-1.673-1.673M12 17l1.673 1.673M16 11V6a2 2 0 00-2-2h-4a2 2 0 00-2 2v5m4 0h.01M12 14H8c-1.1 0-2 .9-2 2v1h12v-1c0-1.1-.9-2-2-2h-4z" />
                </svg>
            </p>
        </div>
        
        <p class="text-gray-600 text-center mb-7 leading-relaxed flex items-center justify-center">
            <svg class="h-5 w-5 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            Enter a URL below to instantly check its legitimacy. Our advanced machine learning model rapidly identifies potential phishing threats.
        </p>

        <form id="checkerForm" class="space-y-5">
            <div class="relative flex items-center">
                <input type="text" id="urlInput" placeholder="e.g., secure-login.example.com" required
                       class="w-full pl-14 pr-28 py-4 rounded-xl border-2 border-gray-300 focus:ring-4 focus:ring-indigo-300 focus:border-indigo-500 transition duration-200 ease-in-out text-lg text-gray-800 placeholder-gray-400 shadow-sm">
                <div class="absolute inset-y-0 left-0 flex items-center pl-4">
                    <svg class="h-7 w-7 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.885a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
                    </svg>
                </div>
                <button type="button" id="pasteButton"
                        class="absolute inset-y-0 right-0 flex items-center pr-4 text-sm font-semibold text-indigo-600 hover:text-indigo-800 transition duration-150 ease-in-out group">
                    <svg class="h-5 w-5 mr-1 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 20l-5.447-2.724A2 2 0 013 15.174V9.826a2 2 0 011.553-1.802L9 6m0 14l6-3m-6 3V6m3 6l3-3m0 0l3-3m-3 3l-3-3m-3 3V6m3 6l3-3m0 0l3-3m-3 3l-3-3"/>
                    </svg>
                    Paste
                </button>
            </div>

            <button type="submit"
                    class="w-full bg-gradient-to-r from-indigo-600 to-purple-700 hover:from-indigo-700 hover:to-purple-800 text-white font-bold py-4 px-6 rounded-xl transition duration-200 ease-in-out transform hover:scale-102 focus:outline-none focus:ring-4 focus:ring-indigo-300 focus:ring-offset-2 disabled:opacity-60 disabled:transform-none shadow-lg">
                <span id="buttonText">Check URL Now</span>
            </button>
        </form>

        <div id="loadingMessage" class="hidden text-center text-indigo-600 mt-8 flex flex-col items-center justify-center">
            <lottie-player src="https://assets7.lottiefiles.com/packages/lf20_tmbg5y4z.json"  background="transparent"  speed="1"  style="width: 80px; height: 80px;"  loop  autoplay></lottie-player>
            <p class="text-lg font-semibold mt-2">Analyzing URL...</p>
        </div>

        <div id="resultBox" class="result-box-transition mt-8 p-6 rounded-xl border-l-4 flex items-center space-x-4 shadow-md">
            <div id="resultIcon" class="flex-shrink-0">
                </div>
            <div>
                <h3 class="text-xl font-bold mb-1" id="resultHeader">Prediction Result:</h3>
                <p id="resultText" class="text-2xl font-extrabold"></p>
                <p id="confidenceText" class="text-sm text-gray-600 mt-1"></p>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('checkerForm');
        const urlInput = document.getElementById('urlInput');
        const pasteButton = document.getElementById('pasteButton');
        const resultBox = document.getElementById('resultBox');
        const resultText = document.getElementById('resultText');
        const resultIcon = document.getElementById('resultIcon');
        const confidenceText = document.getElementById('confidenceText');
        const loadingMessage = document.getElementById('loadingMessage');
        const checkButton = form.querySelector('button');
        const buttonText = document.getElementById('buttonText');

        // This function automatically cleans the URL when the paste button is clicked.
        pasteButton.addEventListener('click', async () => {
            try {
                const text = await navigator.clipboard.readText();
                const cleanedText = text.replace(/^(https?:\/\/)?(www\.)?/, '');
                urlInput.value = cleanedText;
            } catch (err) {
                console.error('Failed to read clipboard contents: ', err);
                showCustomMessage('Failed to paste from clipboard. Please paste manually.', 'bg-red-100 text-red-800 border-red-500', '<svg class="h-10 w-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>', 0, false);
            }
        });
        
        // This function automatically cleans the URL when Ctrl+V is used to paste.
        urlInput.addEventListener('paste', (e) => {
            e.preventDefault(); // Prevents the default paste
            const pastedText = (e.clipboardData || window.clipboardData).getData('text');
            
            // Corrected JavaScript regex to fix the Python SyntaxWarning.
            const regex = new RegExp('^(https?:\/\/)?(www\.)?');
            const cleanedText = pastedText.replace(regex, '');
            
            // Insert the cleaned text at the cursor position
            document.execCommand('insertText', false, cleanedText);
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Clean the URL on form submission
            let url = urlInput.value;
            // Corrected JavaScript regex to fix the Python SyntaxWarning.
            const regex = new RegExp('^(https?:\/\/)?(www\.)?');
            const cleanedUrl = url.replace(regex, '');

            urlInput.value = cleanedUrl; // Update the input field with the cleaned URL

            if (!cleanedUrl) {
                showCustomMessage('Please enter a URL to check.', 'bg-yellow-100 text-yellow-800 border-yellow-500', '', 0, false);
                return;
            }

            loadingMessage.classList.remove('hidden');
            resultBox.classList.remove('active', 'animate-shake');
            resultBox.classList.add('hidden');
            checkButton.disabled = true;
            buttonText.textContent = 'Checking...';
            confidenceText.textContent = '';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: cleanedUrl }), // Use the cleaned URL for prediction
                });

                const data = await response.json();

                if (response.ok) {
                    let messageText;
                    let boxClasses;
                    let iconHtml;

                    if (data.is_phishing) {
                        messageText = `The URL is likely to be ${data.prediction}.`;
                        boxClasses = 'bg-red-100 text-red-800 border-red-500';
                        iconHtml = '<svg class="h-10 w-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
                        showCustomMessage(messageText + `<br><span class="break-all font-normal text-sm opacity-80">${data.url}</span>`, boxClasses, iconHtml, data.confidence, true);

                    } else {
                        messageText = `The URL is likely to be ${data.prediction}. You can visit:`;
                        boxClasses = 'bg-green-100 text-green-800 border-green-500';
                        iconHtml = '<svg class="h-10 w-10 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
                        showCustomMessage(messageText + `<br><a href="${data.url}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline break-all font-normal text-sm">${data.url}</a>`, boxClasses, iconHtml, data.confidence, false);
                    }
                    

                } else {
                    showCustomMessage(`Error: ${data.error}`, 'bg-red-100 text-red-800 border-red-500', '<svg class="h-10 w-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>', 0, false);
                }

            } catch (error) {
                showCustomMessage(`An unexpected error occurred: ${error}`, 'bg-red-100 text-red-800 border-red-500', '<svg class="h-10 w-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>', 0, false);
            } finally {
                loadingMessage.classList.add('hidden');
                checkButton.disabled = false;
                buttonText.textContent = 'Check URL Now';
                resultBox.classList.remove('hidden');
                setTimeout(() => {
                    resultBox.classList.add('active');
                    resultIcon.classList.add('icon-pop-in');
                }, 10);
            }
        });

        function showCustomMessage(messageHtml, classNames, iconSvg, confidence, isPhishing) {
            resultBox.className = `result-box-transition mt-8 p-6 rounded-xl border-l-4 flex items-center space-x-4 shadow-md ${classNames}`;
            resultText.innerHTML = messageHtml;
            resultIcon.innerHTML = iconSvg;
            confidenceText.textContent = confidence > 0 ? `Confidence: ${confidence}%` : '';
            resultIcon.classList.remove('icon-pop-in');
            if (isPhishing) {
                resultBox.classList.add('animate-shake');
            }
        }
    </script>
</body>
</html>
"""
