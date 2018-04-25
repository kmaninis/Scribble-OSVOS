# Scribble-OSVOS

This repository contains a baseline for the interactive track of the [DAVIS Challenge on Video Object Segmentation Wokshop](http://davischallenge.org/challenge2018/index.html) held in CVPR 2018.

This PyTorch code is based on the original [OSVOS-Pytorch](https://github.com/kmaninis/OSVOS-PyTorch) implementation. It adapts the orignal [OSVOS](http://vision.ee.ethz.ch/~cvlsegmentation/osvos) to train only in scribbles instead of the full mask.


### Installation:
The code was tested with [Miniconda](https://conda.io/miniconda.html) and Python 3.6. After installing the Miniconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/kmaninis/Scribble-OSVOS
    cd Scribble-OSVOS
    ```
 
1. Install dependencies:
    ```Shell
    conda install pytorch=0.3.1 torchvision -c pytorch
    conda install matplotlib opencv pillow scikit-learn scikit-image
    ```
3. Install the DAVIS interactive package following [these](http://interactive.davischallenge.org/user_guide/installation/) instructions ('PyPi Install' section), and download the scribbles ('DAVIS Dataset' section). 
  
2. Download the model by running the script inside ```models/```:
    ```Shell
    cd models/
    chmod +x download_osvos_parent.sh
    ./download_osvos_parent.sh
    cd ..
    ```
3. Edit the path to [DAVIS 2017](http://davischallenge.org/davis2017/code.html) in mypath.py

4. Modify any parameters in ``demo_interactive.py`` (for example the gpu_id).

5. To run the interactive session (with the default parameters it takes 10 hours on a Titan Xp):
    ```Shell
    python demo_interactive.py
    ```
6. A CSV report with all the metrics will be generated in ``results``. The expected output after running all sequences can be found at `results/result_default_settings.csv`. `analyze_report.py` will generate an overall report of the results.

Enjoy!

### Citation:
	@Inproceedings{Caelles_arXiv_2018,
	  Title          = {The 2018 DAVIS Challenge on Video Object Segmentation},
	  Author         = {Sergi Caelles and Alberto Montes and Kevis-Kokitsi Maninis and Yuhua Chen and Luc {Van Gool} and Federico Perazzi and Jordi Pont-Tuset},
	  journal        = {arXiv:1803.00557},
	  Year           = {2018}
	}
If you encounter any problems with the code, want to report bugs, etc. please open an issue or contact us at {kmaninis, scaelles}[at]vision[dot]ee[dot]ethz[dot]ch.

