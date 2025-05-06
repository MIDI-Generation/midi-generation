# midi-generation
Midi Generation uses a transformer-based VAE to conditionally generate classical music MIDI files based on a given composer. Overall, the current version of the model has weak label conditioning and generates music that sounds similar regardless of composer. However, the actual sound of this music is not too bad. We recommend trying out a view of the .mid files included in the git.
# Model Architecture
![image](https://github.com/user-attachments/assets/9bb9e6cd-8b0f-4856-a8bf-c0a474364a6e)
We utilize PyTorch's built in Transformer Encoder and Decoder objects to build a Variational Autoencoder to generate MIDI sequences. We then feed that MIDI sequence through a pre-trained GRU classifier to obtain a confidence score. We use CrossEntropy to measure the reconstruction loss, and use KL-Divergence with a mixture of gaussian prior distribution. During inference, all that's required for the decoder is feeding a random gaussian vector with the correct label mean and a label, upon doing this it will generate a full 500 token length MIDI sequence which can be easily saved as a file. The encoder follows traditional VAE architecture, basically appended to the output of a transformer encoder. For the decoder we use both self and cross attention to strengthen the label conditioning.
# Comparing Model Accuracy
![image](https://github.com/user-attachments/assets/42a81f12-50fa-4bdc-9fe1-912bc3a729b5)
The accuracy of our models are almost always below random baseline. We used a subset of 12 classes of our full 34 class dataset.The only model that does beat random baseline is our original model with no LayerNorm. This shows that the generated sequences are very similar regardless of label.
# Comparing Model Relative Cosine Similarity
![image](https://github.com/user-attachments/assets/3bda05a3-1e00-4c26-8d54-fee3aa8a93cc)
The accuracy of our models are almost always below random baseline. Each of the models gives a near-zero relative cosine similarity score, illustrating that our generated sequences are not unique for each label, being just as similar to other classes.
# Visualization of Latent Space
![image](https://github.com/user-attachments/assets/ddce73ab-afb7-4269-af5f-d2c2ee6a384b)
Our latent space shows a moderate amount of separation between classes, with a very noisy bunch within the center of the mean. This means that the classes separated from the rest probably have more unique generated sequences.
# Comparing Model Silhouette Score
![image](https://github.com/user-attachments/assets/6b94ebb8-bad1-4926-9a49-f520e8aacd04)
Our latent space shows a moderate amount of separation between classes, with a 0.2 silhouette score regardless of model. This means that the latent space is closer to being mixed than being cleanly separated, but it is significantly above 0.
# Confusion Matrix of Best Model
![image](https://github.com/user-attachments/assets/09ab8cef-3498-4d80-a851-3fb95a321a5b)
Model loves guessing Frederic Chopin.
Remember to use `git lfs pull` to download the MIDI tensor files (and to make sure you have git lfs installed). A regular `git pull` will not download them.

## Using MIDIVAE_new.ipynb (and variants)

To run MIDIVAE_new.ipynb, we strongly suggest following the below proceedure:
1. Go to the SCC website, click on `Files` and upload the `maestro_new_splits_no_augmentation.csv` ([click here to download it](https://drive.google.com/file/d/1HU-lg5HUxXzuaV72yaIFfDMYe-YD4Tns/view)) and `MIDIVAE.ipynb` files in your choice of directory (or use our project directory: `/projectnb/ec523/projects/proj_MIDIgen`, which already has the files in it)
2. Click on click on `Interactive Apps`, and click on `Jupyter Notebooks`
3. We are going to request and launch an interactive Jupyter Notebook session with the following parameters:
    - List of modules to load (space separated): miniconda academic-ml/spring-2025
    - Pre-Launch Command (optional): conda activate spring-2025-pyt
    - Interface: notebook
    - Working Directory: wherever you uploaded the files in step 1
    - Number of hours: 12
    - Number of cores: 4
    - Number of GPUs: 1
    - GPU compute capability: 8.0
4. Click `launch` and wait for your session to begin
5. Once your session begins, run all cells and a new model will begin training.
6. Once the model is trained, it will automatically save, install prettyMIDI with pip, then run inference to generate a .MID music file
7. To listen to this file, download it and run it using your favorite MIDI synthesizer or convert it to a .WAV with [a free online tool such as this one](https://www.zamzar.com/convert/midi-to-wav/). You can then play the WAV on virtually any music playing app such as VLC or Windows Media Player

Warning: We highly recommend AGAINST running MIDIVAE.ipynb in Google Colab. Google Colab has a different configuration for PyTorch than the SCC's academic-ml/spring-2025 they are NOT cross-compatible. 
