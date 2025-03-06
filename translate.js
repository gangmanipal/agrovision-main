// Google Translate API Key
const apiKey = "C:\Users\Honisha\Downloads\onlineeatsbot-tfdy-e292187eb1c3.json";

// List of languages spoken in India
const languages = {
  'en': 'English',
  'hi': 'Hindi',
  'bn': 'Bengali',
  'te': 'Telugu',
  'mr': 'Marathi',
  'ta': 'Tamil',
  'gu': 'Gujarati',
  'ur': 'Urdu',
  'kn': 'Kannada',
  'ml': 'Malayalam',
  'or': 'Odia',
  'pa': 'Punjabi',
  'as': 'Assamese',
  'ma': 'Maithili',
  'bh': 'Bhojpuri',
  'sa': 'Sanskrit'
};

// Function to translate text
async function translateText(text, targetLang) {
  const url = `https://translation.googleapis.com/language/translate/v2?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({
      q: text,
      target: targetLang
    }),
    headers: {
      'Content-Type': 'application/json'
    }
  });

  const data = await response.json();
  return data.data.translations[0].translatedText;
}

// Function to translate the entire page
async function translatePage(targetLang) {
  const elements = document.querySelectorAll('[data-translate]');
  for (const el of elements) {
    const text = el.innerText;
    const translatedText = await translateText(text, targetLang);
    el.innerText = translatedText;
  }
}

// Function to handle language selection
function handleLanguageChange(event) {
  const selectedLang = event.target.value;
  localStorage.setItem('selectedLanguage', selectedLang);
  translatePage(selectedLang);
}

// Function to initialize the language selection
function initializeLanguageSelection() {
  const languageSelect = document.getElementById('languageSelect');
  languageSelect.addEventListener('change', handleLanguageChange);

  const savedLang = localStorage.getItem('selectedLanguage');
  if (savedLang) {
    languageSelect.value = savedLang;
    translatePage(savedLang);
  }
}

document.addEventListener('DOMContentLoaded', initializeLanguageSelection);