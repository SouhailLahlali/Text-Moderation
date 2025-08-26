# Text Toxicity Detector
A Python application that detects toxicity in text and classifies it as **SFW (Safe For Work)** or **NSFW (Not Safe For Work)**. This project helps identify harmful or inappropriate content in textual data.

---

## Features
- Detect toxic or offensive language in text.
- Classify text as SFW or NSFW.
- Easy-to-use Python interface.
- Can be integrated into chat apps, forums, or moderation tools.

---

## Setup & Running

1. **Download the Model**  
   The model required to run this application is stored on Google Drive. Download it from [this link](https://drive.google.com/drive/u/0/folders/1Rb9gbVuxeRIxz_fSUQifkSkAMJs507Hd).
   
2. **Place the Folder in the Project Directory**  
   After downloading, place the folder in the same directory as `app.py`.

3. **Create a Virtual Environment**  
   It's recommended to use a virtual environment to manage dependencies.

   **For Windows:**
   ```bash
   python -m venv env
   env\Scripts\activate
   ```

   **For Linux/macOS:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

4. **Install Dependencies**  
   Use pip to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```
