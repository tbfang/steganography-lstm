# steganography-nn

Repository containing code, paper, and demo for an LSTM that generates steganographic text.

### Instructions to Launch the Demo

To launch the demo, first `cd` into `demo/stego-lstm`. All the instructions below will be done in that directory.

I suggest creating a virtual environment, so that running this app does not conflict with your system. `pip install virtualenv` to install the tool. Then, assuming you're in `stego-lstm`, type `venv stego` and this creates a virtual environment for this project called "stego".

Install the dependencies (e.g. Flask, Torch, WTForms) with `pip install -r requirements.txt` within the "stego" environment. If installing PyTorch fails through this command, go to pytorch.org and use the command line installer from the home page.

Run the web app by using `python run.py` in the `stego-lstm` folder.