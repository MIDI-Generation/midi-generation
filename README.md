# midi-generation

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
6. Once the model is trained, it will automatically save, then run inference to generate a .MID music file
7. To listen to this file, install prettyMIDI with pip, download it and run it using your favorite MIDI synthesizer or convert it to a .WAV with [a free online tool such as this one](https://www.zamzar.com/convert/midi-to-wav/). You can then play the WAV on virtually any music playing app such as VLC or Windows Media Player

Warning: We highly recommend AGAINST running MIDIVAE.ipynb in Google Colab. Google Colab has a different configuration for PyTorch than the SCC's academic-ml/spring-2025 they are NOT cross-compatible. 
