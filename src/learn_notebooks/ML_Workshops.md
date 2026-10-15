**NOTE:** You are encouraged to experiment with different visualization techniques and quality checks. You can use AI assistants to help troubleshoot issues, explain unfamiliar concepts, or suggest additional analyses.

**AI TOOLS:** ChatGPT, Claude, MetaAI, Poe.com, etc.

---

### **Task 1 (Week 1): Loading, Visualizing, and Quality-Checking EEG Data**

#### **Step 1: Data Acquisition**

1. **Visit the Universum GitHub repository:**

   - [UniversumX/Universum - Data Collection](https://github.com/UniversumX/Universum/tree/main/DataCollection)

Certainly! I'll restructure and clarify this information to make it more clear and easier to understand:
/
2. **Download the EEG data files from the DataCollection folder:**

   a) **Data Location:** 
      - Navigate to the "data" subfolder within the DataCollection folder.
      - This folder will be present if you've run the data collection pipeline.

   b) **Folder Structure:**
      The data is organized in a hierarchical structure:
      ```
      DataCollection/
      └── data/
          └── [participant_number]/
              └── [visit_number]/
                  └── [trial_number]/
      ```
      For example: `DataCollection/data/001/01/001/`

   c) **Data Files:**
      Each trial folder contains two CSV files:
      - `accelerometer_data.csv`
      - `eeg_data_raw.csv`

   d) **File Formats:**
      - `eeg_data_raw.csv`:
        - 9 columns total
        - Column 1: Timestamp
        - Columns 2-9: Data from 8 EEG channels

      - `accelerometer_data.csv`:
        - 9 columns total
        - Column 1: Timestamp
        - Column 2: Device ID
        - Columns 3-5: X, Y, Z (spatial coordinates)
        - Columns 6-7: Pitch and Roll angles (likely in degrees)
        - Column 8: Acceleration
        - Column 9: Inclination
        Note: Exact meanings of accelerometer data columns are not fully specified by Neurosity.

   e) **Alternative Data Format:**
      - If you prefer to work with MNE Raw Array format:
        1. Navigate to the ModelDevelopment folder
        2. Run the `preprocessing.py` script
        3. This will convert the CSV data into MNE Raw Array format

   - Click on each data file and select **"Download"**.
   - Alternatively, clone the repository using `git clone https://github.com/UniversumX/Universum.git`.

3. **Store the data either in your Google Drive or on your local machine.**

#### **Step 2: Setting Up the Environment**

1. **Choose your preferred working environment:**

   - **Google Colab**
   - **Jupyter Notebook**
   - **Code editor** like VS Code with an appropriate Python environment.

2. **If using Google Colab:**

   - Upload the data to your Google Drive.
   - Mount your Google Drive in the Colab notebook.

   **AI Assistant Tip:** If you're unsure how to set up your environment or mount Google Drive in Colab, you can ask:

   *"How do I set up a Python environment for data analysis? Can you show me how to mount Google Drive in a Colab notebook?"*

#### **Step 3: Loading the Data**

1. **Import necessary libraries:**

   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import mne  # If your data is in EDF or other EEG formats
   ```

2. **Load the EEG data:**

   - If your data is in CSV format:

     ```python
     # Adjust the file path as needed
     eeg_data = pd.read_csv('path/to/your/eeg_file.csv')
     ```

   - If your data is in EDF format:

     ```python
     raw = mne.io.read_raw_edf('path/to/your/eeg_file.edf', preload=True)
     ```

   **AI Assistant Tip:** If you encounter issues or need help with specific file formats, you can ask:

   *"How do I load EEG data from a CSV file using Python?"*

   *"How do I load an EEG file with the .edf extension using Python and the MNE library?"*

#### **Step 4: Visualizing the Data**

1. **Visualize EEG signals:**

   - For data loaded with pandas:

     ```python
     # Plot a specific channel
     plt.figure(figsize=(15, 5))
     plt.plot(eeg_data['Channel1'])  # Replace 'Channel1' with your channel name
     plt.title('EEG Signal - Channel1')
     plt.xlabel('Time')
     plt.ylabel('Amplitude')
     plt.show()
     ```

   - For data loaded with MNE:

     ```python
     # Plot raw EEG signals
     raw.plot(scalings='auto', show=True, block=True)
     ```

2. **Generate a Power Spectral Density (PSD) plot:**

   - Using MNE:

     ```python
     raw.plot_psd(fmax=50)  # Adjust fmax as needed
     ```

   **AI Assistant Tip:** To learn more about different visualization techniques, ask:

   *"What are some effective ways to visualize EEG data in Python?"*

   *"Can you show me how to create a topographic map of EEG activity?"*

#### **Step 5: Quality Check**

1. **Check for missing data or anomalies:**

   - For pandas DataFrame:

     ```python
     print(eeg_data.isnull().sum())
     ```

   - For MNE Raw object:

     ```python
     print(raw.info['ch_names'])
     print(raw.get_data().shape)
     ```

2. **Analyze Signal-to-Noise Ratio (SNR):**

   - **Note:** Estimating SNR can be complex; consider using variance or standard deviation as proxies.

   - For MNE:

     ```python
     # This function is not built-in; you'd need to define it or use alternatives
     # Here's a simple way to check signal variance across channels
     import numpy as np
     data = raw.get_data()
     channel_variances = np.var(data, axis=1)
     plt.bar(raw.ch_names, channel_variances)
     plt.xlabel('Channels')
     plt.ylabel('Variance')
     plt.title('Channel Variances')
     plt.show()
     ```

3. **Identify and handle artifacts:**

   - **Detect flat channels:**

     ```python
     flat_channels = mne.preprocessing.find_flat_channels(raw)
     print(f"Flat channels: {flat_channels['flat']}")
     ```

   - **Detect bad segments:**

     ```python
     from mne.preprocessing import annotate_bad_segments
     annotations = annotate_bad_segments(raw, picks='eeg', verbose=True)
     raw.set_annotations(annotations)
     raw.plot()
     ```

   **AI Assistant Tip:** For more advanced quality checks or artifact removal techniques, you can ask:

   *"What are common artifacts in EEG data, and how can I detect and remove them using Python?"*

   *"How can I apply filtering to clean EEG data?"*

#### **Step 6: Documentation and Reporting**

1. **Document your findings and observations:**

   - Note any channels with issues.
   - Describe any anomalies or artifacts found.

2. **Create a brief report summarizing the data quality and any issues encountered.**

   **AI Assistant Tip:** To improve your documentation skills, you might ask:

   *"Can you provide a template or outline for documenting EEG data quality analysis findings?"*

---

### **Task 2 (Week 2): Whitening EEG Data Using PCA**

#### **Step 1: Understanding PCA Whitening**

1. **Principal Component Analysis (PCA)** is a dimensionality reduction technique that can also be used for whitening data.

2. **Whitening** transforms data to have unit variance and removes correlations between features.

   **AI Assistant Tip:** If you need a refresher on PCA or whitening, you can ask:

   *"What is PCA and how does it work?"*

   

   *"How do I perform data whitening using PCA in Python?"*

#### **Step 2: Preprocessing**

1. **Ensure your EEG data is properly formatted:**

   - The data should be in a NumPy array or pandas DataFrame.

2. **Standardize the data:**

   - Remove the mean from each channel.

   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   eeg_data_standardized = scaler.fit_transform(eeg_data)
   ```

#### **Step 3: Applying PCA Whitening**

1. **Import PCA from scikit-learn:**

   ```python
   from sklearn.decomposition import PCA
   ```

2. **Apply PCA with whitening:**

   ```python
   pca = PCA(whiten=True)
   eeg_data_whitened = pca.fit_transform(eeg_data_standardized)
   ```

3. **Inspect the explained variance ratio:**

   ```python
   plt.plot(np.cumsum(pca.explained_variance_ratio_))
   plt.xlabel('Number of Components')
   plt.ylabel('Cumulative Explained Variance')
   plt.title('Explained Variance by PCA Components')
   plt.show()
   ```

   **AI Assistant Tip:** To understand the importance of components, ask:

   *"How many principal components should I keep for my EEG data?"*

   *"What does the explained variance ratio tell me in PCA?"*

#### **Step 4: Saving or Using the Whitened Data**

1. **Save the whitened data for future use:**

   ```python
   np.save('eeg_data_whitened.npy', eeg_data_whitened)
   ```

2. **Proceed to the next task using the whitened data.**

---

### **Task 3 (Week 3): Independent Component Analysis (ICA) on Whitened Data**

#### **Step 1: Understanding ICA**

1. **Independent Component Analysis (ICA)** separates multivariate signals into additive subcomponents that are maximally independent.

   **AI Assistant Tip:** If you need more information on ICA, you can ask:

   *"What is Independent Component Analysis and how is it used in EEG data processing?"*

#### **Step 2: Importing Necessary Libraries**

1. **Use scikit-learn or MNE for ICA:**

   ```python
   from sklearn.decomposition import FastICA
   # or using MNE
   import mne
   ```

#### **Step 3: Applying ICA**

1. **Using scikit-learn's FastICA:**

   ```python
   ica = FastICA(n_components=number_of_components, random_state=0)
   eeg_data_ica = ica.fit_transform(eeg_data_whitened)
   ```

2. **Using MNE's ICA implementation:**

   ```python
   ica = mne.preprocessing.ICA(n_components=number_of_components, random_state=0)
   ica.fit(raw)
   ```

3. **Identify and Remove Artifacts:**

   - **Plot ICA components:**

     ```python
     ica.plot_components()
     ```

   - **Exclude artifact components (e.g., eye blinks, heartbeats):**

     ```python
     ica.exclude = [0, 1]  # Indices of components identified as artifacts
     raw_corrected = raw.copy()
     ica.apply(raw_corrected)
     ```

   **AI Assistant Tip:** To learn how to identify artifacts in ICA components, ask:

   *"How do I identify and remove EOG and ECG artifacts using ICA in EEG data?"*

#### **Step 4: Saving or Using the Cleaned Data**

1. **Save the ICA-cleaned data:**

   - For scikit-learn:

     ```python
     np.save('eeg_data_ica_cleaned.npy', eeg_data_ica)
     ```

   - For MNE:

     ```python
     raw_corrected.save('raw_ica_cleaned.fif', overwrite=True)
     ```

---

### **Task 4 (Week 4): Feature Extraction and Machine Learning Modeling**

#### **Step 1: Preparing the Cleaned Data**

1. **Load the cleaned EEG data from previous steps.**

   ```python
   # For NumPy arrays
   eeg_data_cleaned = np.load('eeg_data_ica_cleaned.npy')

   # For MNE Raw objects
   raw_cleaned = mne.io.read_raw_fif('raw_ica_cleaned.fif', preload=True)
   ```

#### **Step 2: Feature Extraction**

1. **Choose features to extract:**

   - **Spectrograms**
   - **Raw signal segments**
   - **Statistical features (mean, variance, etc.)**
   - **Frequency domain features**

2. **Extract Spectrogram Features:**

   ```python
   from scipy.signal import spectrogram

   f, t, Sxx = spectrogram(eeg_data_cleaned, fs=sampling_rate)
   # fs is the sampling frequency
   ```

3. **Flatten or Reshape Features for Modeling:**

   ```python
   features = Sxx.reshape(Sxx.shape[0], -1)
   ```

   **AI Assistant Tip:** To explore different feature extraction methods, ask:

   *"What are effective feature extraction techniques for EEG data?"*

   *"How do I extract frequency domain features from EEG signals?"*

#### **Step 3: Building Machine Learning Models**

1. **Split the data into training and testing sets:**

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
   ```

   - **Note:** You'll need labels corresponding to your EEG data.

2. **Choose a machine learning model:**

   - **Examples:** Support Vector Machine (SVM), Random Forest, Neural Networks.

3. **Train the Model:**

   ```python
   from sklearn.ensemble import RandomForestClassifier

   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   ```

4. **Evaluate the Model:**

   ```python
   from sklearn.metrics import accuracy_score, classification_report

   y_pred = clf.predict(X_test)
   print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
   print(classification_report(y_test, y_pred))
   ```

   **AI Assistant Tip:** If you need help with model selection or evaluation, ask:

   *"What machine learning models are suitable for EEG classification tasks?"*

   *"How can I evaluate the performance of my EEG classification model?"*

#### **Step 4: Experimentation and Improvement**

1. **Try different models and parameters.**

2. **Perform cross-validation:**

   ```python
   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(clf, features, labels, cv=5)
   print(f"Cross-validation scores: {scores}")
   ```

3. **Tune hyperparameters using Grid Search or Random Search.**

   **AI Assistant Tip:** For guidance on improving your model, ask:

   *"How can I perform hyperparameter tuning for my machine learning model?"*

   *"What techniques can I use to prevent overfitting in my EEG data model?"*

#### **Step 5: Documenting and Reporting**

1. **Document your modeling process and results.**

2. **Create visualizations of model performance.**

   - **Confusion Matrix:**

     ```python
     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

     cm = confusion_matrix(y_test, y_pred)
     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
     disp.plot()
     plt.show()
     ```

3. **Summarize findings and suggest next steps.**

---

**Final Notes:**

- **Collaboration:** Feel free to discuss your approaches and findings with your peers.

- **AI Assistance:** Remember to leverage AI tools whenever you encounter challenges or wish to deepen your understanding.

- **Documentation:** Keep thorough notes and comment your code to make it understandable for others and your future self.

---

**Happy Coding!**

---

*Please feel free to reach out to the Universum Team Leads for any questions or issues you found in the documentation*